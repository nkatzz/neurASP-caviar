
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(mws1gt_1,143,368). coords(mws1gt_1,70,84,368).  visible(mws1gt_1,368). happensAt(walking(mws1gt_1),368). orientation(mws1gt_3,165,368). coords(mws1gt_3,46,159,368).  visible(mws1gt_3,368). happensAt(walking(mws1gt_3),368).
orientation(mws1gt_1,143,369). coords(mws1gt_1,70,84,369).  visible(mws1gt_1,369). happensAt(walking(mws1gt_1),369). orientation(mws1gt_3,165,369). coords(mws1gt_3,45,160,369).  visible(mws1gt_3,369). happensAt(walking(mws1gt_3),369).
orientation(mws1gt_1,143,370). coords(mws1gt_1,70,84,370).  visible(mws1gt_1,370). happensAt(walking(mws1gt_1),370). orientation(mws1gt_3,165,370). coords(mws1gt_3,45,160,370).  visible(mws1gt_3,370). happensAt(walking(mws1gt_3),370).
orientation(mws1gt_1,143,371). coords(mws1gt_1,70,82,371).  visible(mws1gt_1,371). happensAt(walking(mws1gt_1),371). orientation(mws1gt_3,165,371). coords(mws1gt_3,44,160,371).  visible(mws1gt_3,371). happensAt(walking(mws1gt_3),371).
orientation(mws1gt_1,143,372). coords(mws1gt_1,70,81,372).  visible(mws1gt_1,372). happensAt(walking(mws1gt_1),372). orientation(mws1gt_3,165,372). coords(mws1gt_3,44,160,372).  visible(mws1gt_3,372). happensAt(walking(mws1gt_3),372).
orientation(mws1gt_1,143,373). coords(mws1gt_1,70,80,373).  visible(mws1gt_1,373). happensAt(walking(mws1gt_1),373). orientation(mws1gt_3,165,373). coords(mws1gt_3,44,160,373).  visible(mws1gt_3,373). happensAt(walking(mws1gt_3),373).
orientation(mws1gt_1,143,374). coords(mws1gt_1,70,79,374).  visible(mws1gt_1,374). happensAt(walking(mws1gt_1),374). orientation(mws1gt_3,165,374). coords(mws1gt_3,43,160,374).  visible(mws1gt_3,374). happensAt(walking(mws1gt_3),374).
orientation(mws1gt_1,143,375). coords(mws1gt_1,70,78,375).  visible(mws1gt_1,375). happensAt(walking(mws1gt_1),375). orientation(mws1gt_3,165,375). coords(mws1gt_3,42,160,375).  visible(mws1gt_3,375). happensAt(walking(mws1gt_3),375).
orientation(mws1gt_1,143,376). coords(mws1gt_1,70,77,376).  visible(mws1gt_1,376). happensAt(walking(mws1gt_1),376). orientation(mws1gt_3,165,376). coords(mws1gt_3,42,160,376).  visible(mws1gt_3,376). happensAt(walking(mws1gt_3),376).
orientation(mws1gt_1,143,377). coords(mws1gt_1,70,77,377).  visible(mws1gt_1,377). happensAt(walking(mws1gt_1),377). orientation(mws1gt_3,165,377). coords(mws1gt_3,42,160,377).  visible(mws1gt_3,377). happensAt(walking(mws1gt_3),377).
orientation(mws1gt_1,143,378). coords(mws1gt_1,70,76,378).  visible(mws1gt_1,378). happensAt(walking(mws1gt_1),378). orientation(mws1gt_3,165,378). coords(mws1gt_3,42,160,378).  visible(mws1gt_3,378). happensAt(walking(mws1gt_3),378).
orientation(mws1gt_1,143,379). coords(mws1gt_1,70,76,379).  visible(mws1gt_1,379). happensAt(walking(mws1gt_1),379). orientation(mws1gt_3,165,379). coords(mws1gt_3,42,160,379).  visible(mws1gt_3,379). happensAt(walking(mws1gt_3),379).
orientation(mws1gt_1,143,380). coords(mws1gt_1,70,76,380).  visible(mws1gt_1,380). happensAt(walking(mws1gt_1),380). orientation(mws1gt_3,165,380). coords(mws1gt_3,42,160,380).  visible(mws1gt_3,380). happensAt(walking(mws1gt_3),380).
orientation(mws1gt_1,143,381). coords(mws1gt_1,70,76,381).  visible(mws1gt_1,381). happensAt(walking(mws1gt_1),381). orientation(mws1gt_3,165,381). coords(mws1gt_3,42,161,381).  visible(mws1gt_3,381). happensAt(walking(mws1gt_3),381).
orientation(mws1gt_1,143,382). coords(mws1gt_1,71,76,382).  visible(mws1gt_1,382). happensAt(walking(mws1gt_1),382). orientation(mws1gt_3,165,382). coords(mws1gt_3,41,161,382).  visible(mws1gt_3,382). happensAt(walking(mws1gt_3),382).
orientation(mws1gt_1,143,383). coords(mws1gt_1,71,75,383).  visible(mws1gt_1,383). happensAt(walking(mws1gt_1),383). orientation(mws1gt_3,165,383). coords(mws1gt_3,41,161,383).  visible(mws1gt_3,383). happensAt(walking(mws1gt_3),383).
orientation(mws1gt_1,143,384). coords(mws1gt_1,71,75,384).  visible(mws1gt_1,384). happensAt(walking(mws1gt_1),384). orientation(mws1gt_3,165,384). coords(mws1gt_3,40,161,384).  visible(mws1gt_3,384). happensAt(walking(mws1gt_3),384).
orientation(mws1gt_1,143,385). coords(mws1gt_1,71,75,385).  visible(mws1gt_1,385). happensAt(walking(mws1gt_1),385). orientation(mws1gt_3,165,385). coords(mws1gt_3,39,161,385).  visible(mws1gt_3,385). happensAt(walking(mws1gt_3),385).
orientation(mws1gt_1,143,386). coords(mws1gt_1,71,74,386).  visible(mws1gt_1,386). happensAt(walking(mws1gt_1),386). orientation(mws1gt_3,165,386). coords(mws1gt_3,39,161,386).  visible(mws1gt_3,386). happensAt(walking(mws1gt_3),386).
orientation(mws1gt_1,143,387). coords(mws1gt_1,72,72,387).  visible(mws1gt_1,387). happensAt(walking(mws1gt_1),387). orientation(mws1gt_3,165,387). coords(mws1gt_3,39,161,387).  visible(mws1gt_3,387). happensAt(walking(mws1gt_3),387).
orientation(mws1gt_1,143,388). coords(mws1gt_1,72,71,388).  visible(mws1gt_1,388). happensAt(walking(mws1gt_1),388). orientation(mws1gt_3,165,388). coords(mws1gt_3,39,161,388).  visible(mws1gt_3,388). happensAt(walking(mws1gt_3),388).
orientation(mws1gt_1,143,389). coords(mws1gt_1,72,71,389).  visible(mws1gt_1,389). happensAt(walking(mws1gt_1),389). orientation(mws1gt_3,165,389). coords(mws1gt_3,39,161,389).  visible(mws1gt_3,389). happensAt(walking(mws1gt_3),389).
orientation(mws1gt_1,143,390). coords(mws1gt_1,72,70,390).  visible(mws1gt_1,390). happensAt(walking(mws1gt_1),390). orientation(mws1gt_3,165,390). coords(mws1gt_3,38,161,390).  visible(mws1gt_3,390). happensAt(walking(mws1gt_3),390).
orientation(mws1gt_1,143,391). coords(mws1gt_1,72,69,391).  visible(mws1gt_1,391). happensAt(walking(mws1gt_1),391). orientation(mws1gt_3,165,391). coords(mws1gt_3,38,161,391).  visible(mws1gt_3,391). happensAt(walking(mws1gt_3),391).
orientation(mws1gt_1,143,392). coords(mws1gt_1,72,68,392).  visible(mws1gt_1,392). happensAt(walking(mws1gt_1),392). orientation(mws1gt_3,165,392). coords(mws1gt_3,38,161,392).  visible(mws1gt_3,392). happensAt(walking(mws1gt_3),392).
#show holdsAt/2.