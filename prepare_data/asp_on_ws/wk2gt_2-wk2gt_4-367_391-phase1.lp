
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

	orientation(wk2gt_2,141,367). coords(wk2gt_2,98,63,367).  visible(wk2gt_2,367). happensAt(inactive(wk2gt_2),367). orientation(wk2gt_4,151,367). coords(wk2gt_4,174,140,367).  visible(wk2gt_4,367). happensAt(walking(wk2gt_4),367).
orientation(wk2gt_2,141,368). coords(wk2gt_2,98,63,368).  visible(wk2gt_2,368). happensAt(inactive(wk2gt_2),368). orientation(wk2gt_4,151,368). coords(wk2gt_4,171,139,368).  visible(wk2gt_4,368). happensAt(walking(wk2gt_4),368).
orientation(wk2gt_2,141,369). coords(wk2gt_2,98,63,369).  visible(wk2gt_2,369). happensAt(inactive(wk2gt_2),369). orientation(wk2gt_4,151,369). coords(wk2gt_4,169,137,369).  visible(wk2gt_4,369). happensAt(walking(wk2gt_4),369).
orientation(wk2gt_2,141,370). coords(wk2gt_2,98,63,370).  visible(wk2gt_2,370). happensAt(inactive(wk2gt_2),370). orientation(wk2gt_4,151,370). coords(wk2gt_4,167,136,370).  visible(wk2gt_4,370). happensAt(walking(wk2gt_4),370).
orientation(wk2gt_2,141,371). coords(wk2gt_2,98,63,371).  visible(wk2gt_2,371). happensAt(inactive(wk2gt_2),371). orientation(wk2gt_4,151,371). coords(wk2gt_4,167,135,371).  visible(wk2gt_4,371). happensAt(walking(wk2gt_4),371).
orientation(wk2gt_2,141,372). coords(wk2gt_2,98,63,372).  visible(wk2gt_2,372). happensAt(inactive(wk2gt_2),372). orientation(wk2gt_4,151,372). coords(wk2gt_4,166,135,372).  visible(wk2gt_4,372). happensAt(walking(wk2gt_4),372).
orientation(wk2gt_2,141,373). coords(wk2gt_2,98,63,373).  visible(wk2gt_2,373). happensAt(inactive(wk2gt_2),373). orientation(wk2gt_4,151,373). coords(wk2gt_4,166,134,373).  visible(wk2gt_4,373). happensAt(walking(wk2gt_4),373).
orientation(wk2gt_2,141,374). coords(wk2gt_2,98,63,374).  visible(wk2gt_2,374). happensAt(inactive(wk2gt_2),374). orientation(wk2gt_4,151,374). coords(wk2gt_4,165,134,374).  visible(wk2gt_4,374). happensAt(walking(wk2gt_4),374).
orientation(wk2gt_2,141,375). coords(wk2gt_2,98,63,375).  visible(wk2gt_2,375). happensAt(inactive(wk2gt_2),375). orientation(wk2gt_4,151,375). coords(wk2gt_4,164,133,375).  visible(wk2gt_4,375). happensAt(walking(wk2gt_4),375).
orientation(wk2gt_2,141,376). coords(wk2gt_2,98,63,376).  visible(wk2gt_2,376). happensAt(inactive(wk2gt_2),376). orientation(wk2gt_4,151,376). coords(wk2gt_4,163,133,376).  visible(wk2gt_4,376). happensAt(walking(wk2gt_4),376).
orientation(wk2gt_2,141,377). coords(wk2gt_2,98,63,377).  visible(wk2gt_2,377). happensAt(inactive(wk2gt_2),377). orientation(wk2gt_4,147,377). coords(wk2gt_4,161,131,377).  visible(wk2gt_4,377). happensAt(walking(wk2gt_4),377).
orientation(wk2gt_2,141,378). coords(wk2gt_2,98,63,378).  visible(wk2gt_2,378). happensAt(inactive(wk2gt_2),378). orientation(wk2gt_4,147,378). coords(wk2gt_4,159,130,378).  visible(wk2gt_4,378). happensAt(walking(wk2gt_4),378).
orientation(wk2gt_2,141,379). coords(wk2gt_2,98,63,379).  visible(wk2gt_2,379). happensAt(inactive(wk2gt_2),379). orientation(wk2gt_4,147,379). coords(wk2gt_4,156,128,379).  visible(wk2gt_4,379). happensAt(walking(wk2gt_4),379).
orientation(wk2gt_2,141,380). coords(wk2gt_2,98,63,380).  visible(wk2gt_2,380). happensAt(inactive(wk2gt_2),380). orientation(wk2gt_4,147,380). coords(wk2gt_4,155,126,380).  visible(wk2gt_4,380). happensAt(walking(wk2gt_4),380).
orientation(wk2gt_2,141,381). coords(wk2gt_2,98,63,381).  visible(wk2gt_2,381). happensAt(inactive(wk2gt_2),381). orientation(wk2gt_4,147,381). coords(wk2gt_4,153,125,381).  visible(wk2gt_4,381). happensAt(walking(wk2gt_4),381).
orientation(wk2gt_2,141,382). coords(wk2gt_2,98,63,382).  visible(wk2gt_2,382). happensAt(inactive(wk2gt_2),382). orientation(wk2gt_4,147,382). coords(wk2gt_4,153,124,382).  visible(wk2gt_4,382). happensAt(walking(wk2gt_4),382).
orientation(wk2gt_2,141,383). coords(wk2gt_2,98,63,383).  visible(wk2gt_2,383). happensAt(inactive(wk2gt_2),383). orientation(wk2gt_4,147,383). coords(wk2gt_4,152,122,383).  visible(wk2gt_4,383). happensAt(walking(wk2gt_4),383).
orientation(wk2gt_2,141,384). coords(wk2gt_2,98,63,384).  visible(wk2gt_2,384). happensAt(inactive(wk2gt_2),384). orientation(wk2gt_4,147,384). coords(wk2gt_4,152,122,384).  visible(wk2gt_4,384). happensAt(walking(wk2gt_4),384).
orientation(wk2gt_2,141,385). coords(wk2gt_2,98,63,385).  visible(wk2gt_2,385). happensAt(inactive(wk2gt_2),385). orientation(wk2gt_4,147,385). coords(wk2gt_4,151,121,385).  visible(wk2gt_4,385). happensAt(walking(wk2gt_4),385).
orientation(wk2gt_2,141,386). coords(wk2gt_2,98,63,386).  visible(wk2gt_2,386). happensAt(inactive(wk2gt_2),386). orientation(wk2gt_4,147,386). coords(wk2gt_4,150,121,386).  visible(wk2gt_4,386). happensAt(walking(wk2gt_4),386).
orientation(wk2gt_2,141,387). coords(wk2gt_2,98,63,387).  visible(wk2gt_2,387). happensAt(inactive(wk2gt_2),387). orientation(wk2gt_4,149,387). coords(wk2gt_4,150,121,387).  visible(wk2gt_4,387). happensAt(walking(wk2gt_4),387).
orientation(wk2gt_2,141,388). coords(wk2gt_2,98,63,388).  visible(wk2gt_2,388). happensAt(inactive(wk2gt_2),388). orientation(wk2gt_4,149,388). coords(wk2gt_4,149,120,388).  visible(wk2gt_4,388). happensAt(walking(wk2gt_4),388).
orientation(wk2gt_2,141,389). coords(wk2gt_2,98,63,389).  visible(wk2gt_2,389). happensAt(inactive(wk2gt_2),389). orientation(wk2gt_4,149,389). coords(wk2gt_4,148,119,389).  visible(wk2gt_4,389). happensAt(walking(wk2gt_4),389).
orientation(wk2gt_2,141,390). coords(wk2gt_2,98,63,390).  visible(wk2gt_2,390). happensAt(inactive(wk2gt_2),390). orientation(wk2gt_4,149,390). coords(wk2gt_4,146,118,390).  visible(wk2gt_4,390). happensAt(walking(wk2gt_4),390).
orientation(wk2gt_2,141,391). coords(wk2gt_2,98,63,391).  visible(wk2gt_2,391). happensAt(inactive(wk2gt_2),391). orientation(wk2gt_4,149,391). coords(wk2gt_4,144,117,391).  visible(wk2gt_4,391). happensAt(walking(wk2gt_4),391).
#show holdsAt/2.