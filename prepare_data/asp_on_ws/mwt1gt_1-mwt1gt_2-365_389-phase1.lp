
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

	orientation(mwt1gt_1,160,365). coords(mwt1gt_1,109,144,365).  visible(mwt1gt_1,365). happensAt(walking(mwt1gt_1),365). orientation(mwt1gt_2,140,365). coords(mwt1gt_2,116,124,365).  visible(mwt1gt_2,365). happensAt(walking(mwt1gt_2),365).
orientation(mwt1gt_1,160,366). coords(mwt1gt_1,107,141,366).  visible(mwt1gt_1,366). happensAt(walking(mwt1gt_1),366). orientation(mwt1gt_2,140,366). coords(mwt1gt_2,113,115,366).  visible(mwt1gt_2,366). happensAt(walking(mwt1gt_2),366).
orientation(mwt1gt_1,160,367). coords(mwt1gt_1,106,140,367).  visible(mwt1gt_1,367). happensAt(walking(mwt1gt_1),367). orientation(mwt1gt_2,140,367). coords(mwt1gt_2,113,115,367).  visible(mwt1gt_2,367). happensAt(walking(mwt1gt_2),367).
orientation(mwt1gt_1,160,368). coords(mwt1gt_1,106,139,368).  visible(mwt1gt_1,368). happensAt(walking(mwt1gt_1),368). orientation(mwt1gt_2,140,368). coords(mwt1gt_2,110,115,368).  visible(mwt1gt_2,368). happensAt(walking(mwt1gt_2),368).
orientation(mwt1gt_1,160,369). coords(mwt1gt_1,106,139,369).  visible(mwt1gt_1,369). happensAt(walking(mwt1gt_1),369). orientation(mwt1gt_2,140,369). coords(mwt1gt_2,110,115,369).  visible(mwt1gt_2,369). happensAt(walking(mwt1gt_2),369).
orientation(mwt1gt_1,160,370). coords(mwt1gt_1,105,136,370).  visible(mwt1gt_1,370). happensAt(walking(mwt1gt_1),370). orientation(mwt1gt_2,140,370). coords(mwt1gt_2,110,115,370).  visible(mwt1gt_2,370). happensAt(walking(mwt1gt_2),370).
orientation(mwt1gt_1,160,371). coords(mwt1gt_1,105,136,371).  visible(mwt1gt_1,371). happensAt(walking(mwt1gt_1),371). orientation(mwt1gt_2,140,371). coords(mwt1gt_2,110,115,371).  visible(mwt1gt_2,371). happensAt(walking(mwt1gt_2),371).
orientation(mwt1gt_1,160,372). coords(mwt1gt_1,105,135,372).  visible(mwt1gt_1,372). happensAt(walking(mwt1gt_1),372). orientation(mwt1gt_2,140,372). coords(mwt1gt_2,107,115,372).  visible(mwt1gt_2,372). happensAt(walking(mwt1gt_2),372).
orientation(mwt1gt_1,160,373). coords(mwt1gt_1,105,135,373).  visible(mwt1gt_1,373). happensAt(walking(mwt1gt_1),373). orientation(mwt1gt_2,140,373). coords(mwt1gt_2,107,115,373).  visible(mwt1gt_2,373). happensAt(walking(mwt1gt_2),373).
orientation(mwt1gt_1,160,374). coords(mwt1gt_1,105,135,374).  visible(mwt1gt_1,374). happensAt(walking(mwt1gt_1),374). orientation(mwt1gt_2,140,374). coords(mwt1gt_2,107,115,374).  visible(mwt1gt_2,374). happensAt(walking(mwt1gt_2),374).
orientation(mwt1gt_1,160,375). coords(mwt1gt_1,105,135,375).  visible(mwt1gt_1,375). happensAt(walking(mwt1gt_1),375). orientation(mwt1gt_2,140,375). coords(mwt1gt_2,106,115,375).  visible(mwt1gt_2,375). happensAt(walking(mwt1gt_2),375).
orientation(mwt1gt_1,160,376). coords(mwt1gt_1,105,135,376).  visible(mwt1gt_1,376). happensAt(walking(mwt1gt_1),376). orientation(mwt1gt_2,140,376). coords(mwt1gt_2,105,115,376).  visible(mwt1gt_2,376). happensAt(walking(mwt1gt_2),376).
orientation(mwt1gt_1,160,377). coords(mwt1gt_1,105,135,377).  visible(mwt1gt_1,377). happensAt(walking(mwt1gt_1),377). orientation(mwt1gt_2,140,377). coords(mwt1gt_2,105,114,377).  visible(mwt1gt_2,377). happensAt(walking(mwt1gt_2),377).
orientation(mwt1gt_1,160,378). coords(mwt1gt_1,103,132,378).  visible(mwt1gt_1,378). happensAt(walking(mwt1gt_1),378). orientation(mwt1gt_2,140,378). coords(mwt1gt_2,105,114,378).  visible(mwt1gt_2,378). happensAt(walking(mwt1gt_2),378).
orientation(mwt1gt_1,160,379). coords(mwt1gt_1,103,132,379).  visible(mwt1gt_1,379). happensAt(walking(mwt1gt_1),379). orientation(mwt1gt_2,140,379). coords(mwt1gt_2,105,114,379).  visible(mwt1gt_2,379). happensAt(walking(mwt1gt_2),379).
orientation(mwt1gt_1,160,380). coords(mwt1gt_1,100,130,380).  visible(mwt1gt_1,380). happensAt(walking(mwt1gt_1),380). orientation(mwt1gt_2,140,380). coords(mwt1gt_2,103,113,380).  visible(mwt1gt_2,380). happensAt(walking(mwt1gt_2),380).
orientation(mwt1gt_1,160,381). coords(mwt1gt_1,100,129,381).  visible(mwt1gt_1,381). happensAt(walking(mwt1gt_1),381). orientation(mwt1gt_2,140,381). coords(mwt1gt_2,103,113,381).  visible(mwt1gt_2,381). happensAt(walking(mwt1gt_2),381).
orientation(mwt1gt_1,160,382). coords(mwt1gt_1,100,129,382).  visible(mwt1gt_1,382). happensAt(walking(mwt1gt_1),382). orientation(mwt1gt_2,140,382). coords(mwt1gt_2,101,111,382).  visible(mwt1gt_2,382). happensAt(walking(mwt1gt_2),382).
orientation(mwt1gt_1,160,383). coords(mwt1gt_1,100,129,383).  visible(mwt1gt_1,383). happensAt(walking(mwt1gt_1),383). orientation(mwt1gt_2,140,383). coords(mwt1gt_2,101,111,383).  visible(mwt1gt_2,383). happensAt(walking(mwt1gt_2),383).
orientation(mwt1gt_1,160,384). coords(mwt1gt_1,100,129,384).  visible(mwt1gt_1,384). happensAt(walking(mwt1gt_1),384). orientation(mwt1gt_2,140,384). coords(mwt1gt_2,101,111,384).  visible(mwt1gt_2,384). happensAt(walking(mwt1gt_2),384).
orientation(mwt1gt_1,160,385). coords(mwt1gt_1,97,126,385).  visible(mwt1gt_1,385). happensAt(walking(mwt1gt_1),385). orientation(mwt1gt_2,140,385). coords(mwt1gt_2,98,109,385).  visible(mwt1gt_2,385). happensAt(walking(mwt1gt_2),385).
orientation(mwt1gt_1,160,386). coords(mwt1gt_1,97,126,386).  visible(mwt1gt_1,386). happensAt(walking(mwt1gt_1),386). orientation(mwt1gt_2,140,386). coords(mwt1gt_2,98,109,386).  visible(mwt1gt_2,386). happensAt(walking(mwt1gt_2),386).
orientation(mwt1gt_1,160,387). coords(mwt1gt_1,97,126,387).  visible(mwt1gt_1,387). happensAt(walking(mwt1gt_1),387). orientation(mwt1gt_2,140,387). coords(mwt1gt_2,98,109,387).  visible(mwt1gt_2,387). happensAt(walking(mwt1gt_2),387).
orientation(mwt1gt_1,160,388). coords(mwt1gt_1,96,126,388).  visible(mwt1gt_1,388). happensAt(walking(mwt1gt_1),388). orientation(mwt1gt_2,140,388). coords(mwt1gt_2,97,107,388).  visible(mwt1gt_2,388). happensAt(walking(mwt1gt_2),388).
orientation(mwt1gt_1,160,389). coords(mwt1gt_1,96,126,389).  visible(mwt1gt_1,389). happensAt(walking(mwt1gt_1),389). orientation(mwt1gt_2,140,389). coords(mwt1gt_2,97,107,389).  visible(mwt1gt_2,389). happensAt(walking(mwt1gt_2),389).
#show holdsAt/2.