
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

	orientation(br1gt_2,169,370). coords(br1gt_2,86,207,370).  visible(br1gt_2,370). happensAt(inactive(br1gt_2),370). orientation(br1gt_3,162,370). coords(br1gt_3,34,159,370).  visible(br1gt_3,370). happensAt(walking(br1gt_3),370).
orientation(br1gt_2,169,371). coords(br1gt_2,86,207,371).  visible(br1gt_2,371). happensAt(inactive(br1gt_2),371). orientation(br1gt_3,162,371). coords(br1gt_3,34,159,371).  visible(br1gt_3,371). happensAt(walking(br1gt_3),371).
orientation(br1gt_2,169,372). coords(br1gt_2,86,207,372).  visible(br1gt_2,372). happensAt(inactive(br1gt_2),372). orientation(br1gt_3,162,372). coords(br1gt_3,34,159,372).  visible(br1gt_3,372). happensAt(walking(br1gt_3),372).
orientation(br1gt_2,169,373). coords(br1gt_2,86,207,373).  visible(br1gt_2,373). happensAt(inactive(br1gt_2),373). orientation(br1gt_3,162,373). coords(br1gt_3,34,159,373).  visible(br1gt_3,373). happensAt(walking(br1gt_3),373).
orientation(br1gt_2,169,374). coords(br1gt_2,86,207,374).  visible(br1gt_2,374). happensAt(inactive(br1gt_2),374). orientation(br1gt_3,162,374). coords(br1gt_3,35,159,374).  visible(br1gt_3,374). happensAt(walking(br1gt_3),374).
orientation(br1gt_2,169,375). coords(br1gt_2,86,208,375).  visible(br1gt_2,375). happensAt(inactive(br1gt_2),375). orientation(br1gt_3,162,375). coords(br1gt_3,35,159,375).  visible(br1gt_3,375). happensAt(walking(br1gt_3),375).
orientation(br1gt_2,169,376). coords(br1gt_2,86,208,376).  visible(br1gt_2,376). happensAt(inactive(br1gt_2),376). orientation(br1gt_3,162,376). coords(br1gt_3,35,159,376).  visible(br1gt_3,376). happensAt(walking(br1gt_3),376).
orientation(br1gt_2,169,377). coords(br1gt_2,86,208,377).  visible(br1gt_2,377). happensAt(inactive(br1gt_2),377). orientation(br1gt_3,162,377). coords(br1gt_3,35,159,377).  visible(br1gt_3,377). happensAt(walking(br1gt_3),377).
orientation(br1gt_2,169,378). coords(br1gt_2,86,208,378).  visible(br1gt_2,378). happensAt(inactive(br1gt_2),378). orientation(br1gt_3,162,378). coords(br1gt_3,35,159,378).  visible(br1gt_3,378). happensAt(walking(br1gt_3),378).
orientation(br1gt_2,169,379). coords(br1gt_2,86,208,379).  visible(br1gt_2,379). happensAt(inactive(br1gt_2),379). orientation(br1gt_3,162,379). coords(br1gt_3,35,159,379).  visible(br1gt_3,379). happensAt(walking(br1gt_3),379).
orientation(br1gt_2,169,380). coords(br1gt_2,86,208,380).  visible(br1gt_2,380). happensAt(inactive(br1gt_2),380). orientation(br1gt_3,162,380). coords(br1gt_3,35,159,380).  visible(br1gt_3,380). happensAt(walking(br1gt_3),380).
orientation(br1gt_2,169,381). coords(br1gt_2,86,208,381).  visible(br1gt_2,381). happensAt(inactive(br1gt_2),381). orientation(br1gt_3,162,381). coords(br1gt_3,35,159,381).  visible(br1gt_3,381). happensAt(walking(br1gt_3),381).
orientation(br1gt_2,169,382). coords(br1gt_2,86,208,382).  visible(br1gt_2,382). happensAt(inactive(br1gt_2),382). orientation(br1gt_3,162,382). coords(br1gt_3,37,159,382).  visible(br1gt_3,382). happensAt(walking(br1gt_3),382).
orientation(br1gt_2,169,383). coords(br1gt_2,86,208,383).  visible(br1gt_2,383). happensAt(inactive(br1gt_2),383). orientation(br1gt_3,162,383). coords(br1gt_3,37,159,383).  visible(br1gt_3,383). happensAt(walking(br1gt_3),383).
orientation(br1gt_2,174,384). coords(br1gt_2,86,208,384).  visible(br1gt_2,384). happensAt(inactive(br1gt_2),384). orientation(br1gt_3,162,384). coords(br1gt_3,37,158,384).  visible(br1gt_3,384). happensAt(walking(br1gt_3),384).
orientation(br1gt_2,174,385). coords(br1gt_2,86,208,385).  visible(br1gt_2,385). happensAt(inactive(br1gt_2),385). orientation(br1gt_3,162,385). coords(br1gt_3,38,158,385).  visible(br1gt_3,385). happensAt(walking(br1gt_3),385).
orientation(br1gt_2,174,386). coords(br1gt_2,86,208,386).  visible(br1gt_2,386). happensAt(inactive(br1gt_2),386). orientation(br1gt_3,162,386). coords(br1gt_3,39,158,386).  visible(br1gt_3,386). happensAt(walking(br1gt_3),386).
orientation(br1gt_2,174,387). coords(br1gt_2,86,208,387).  visible(br1gt_2,387). happensAt(inactive(br1gt_2),387). orientation(br1gt_3,162,387). coords(br1gt_3,39,158,387).  visible(br1gt_3,387). happensAt(walking(br1gt_3),387).
orientation(br1gt_2,174,388). coords(br1gt_2,86,208,388).  visible(br1gt_2,388). happensAt(inactive(br1gt_2),388). orientation(br1gt_3,162,388). coords(br1gt_3,41,158,388).  visible(br1gt_3,388). happensAt(walking(br1gt_3),388).
orientation(br1gt_2,174,389). coords(br1gt_2,86,208,389).  visible(br1gt_2,389). happensAt(inactive(br1gt_2),389). orientation(br1gt_3,162,389). coords(br1gt_3,41,158,389).  visible(br1gt_3,389). happensAt(walking(br1gt_3),389).
orientation(br1gt_2,174,390). coords(br1gt_2,86,208,390).  visible(br1gt_2,390). happensAt(inactive(br1gt_2),390). orientation(br1gt_3,162,390). coords(br1gt_3,41,158,390).  visible(br1gt_3,390). happensAt(walking(br1gt_3),390).
orientation(br1gt_2,174,391). coords(br1gt_2,86,208,391).  visible(br1gt_2,391). happensAt(inactive(br1gt_2),391). orientation(br1gt_3,162,391). coords(br1gt_3,41,157,391).  visible(br1gt_3,391). happensAt(walking(br1gt_3),391).
orientation(br1gt_2,7,392). coords(br1gt_2,86,208,392).  visible(br1gt_2,392). happensAt(inactive(br1gt_2),392). orientation(br1gt_3,162,392). coords(br1gt_3,41,157,392).  visible(br1gt_3,392). happensAt(walking(br1gt_3),392).
orientation(br1gt_2,7,393). coords(br1gt_2,86,208,393).  visible(br1gt_2,393). happensAt(inactive(br1gt_2),393). orientation(br1gt_3,162,393). coords(br1gt_3,41,157,393).  visible(br1gt_3,393). happensAt(walking(br1gt_3),393).
orientation(br1gt_2,7,394). coords(br1gt_2,86,208,394).  visible(br1gt_2,394). happensAt(inactive(br1gt_2),394). orientation(br1gt_3,162,394). coords(br1gt_3,41,157,394).  visible(br1gt_3,394). happensAt(walking(br1gt_3),394).
#show holdsAt/2.