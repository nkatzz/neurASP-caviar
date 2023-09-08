
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

	orientation(wk2gt_1,125,367). coords(wk2gt_1,115,53,367).  visible(wk2gt_1,367). happensAt(inactive(wk2gt_1),367). orientation(wk2gt_4,151,367). coords(wk2gt_4,174,140,367).  visible(wk2gt_4,367). happensAt(walking(wk2gt_4),367).
orientation(wk2gt_1,125,368). coords(wk2gt_1,115,53,368).  visible(wk2gt_1,368). happensAt(inactive(wk2gt_1),368). orientation(wk2gt_4,151,368). coords(wk2gt_4,171,139,368).  visible(wk2gt_4,368). happensAt(walking(wk2gt_4),368).
orientation(wk2gt_1,125,369). coords(wk2gt_1,115,53,369).  visible(wk2gt_1,369). happensAt(inactive(wk2gt_1),369). orientation(wk2gt_4,151,369). coords(wk2gt_4,169,137,369).  visible(wk2gt_4,369). happensAt(walking(wk2gt_4),369).
orientation(wk2gt_1,125,370). coords(wk2gt_1,115,53,370).  visible(wk2gt_1,370). happensAt(inactive(wk2gt_1),370). orientation(wk2gt_4,151,370). coords(wk2gt_4,167,136,370).  visible(wk2gt_4,370). happensAt(walking(wk2gt_4),370).
orientation(wk2gt_1,125,371). coords(wk2gt_1,115,53,371).  visible(wk2gt_1,371). happensAt(inactive(wk2gt_1),371). orientation(wk2gt_4,151,371). coords(wk2gt_4,167,135,371).  visible(wk2gt_4,371). happensAt(walking(wk2gt_4),371).
orientation(wk2gt_1,125,372). coords(wk2gt_1,115,53,372).  visible(wk2gt_1,372). happensAt(inactive(wk2gt_1),372). orientation(wk2gt_4,151,372). coords(wk2gt_4,166,135,372).  visible(wk2gt_4,372). happensAt(walking(wk2gt_4),372).
orientation(wk2gt_1,125,373). coords(wk2gt_1,115,53,373).  visible(wk2gt_1,373). happensAt(inactive(wk2gt_1),373). orientation(wk2gt_4,151,373). coords(wk2gt_4,166,134,373).  visible(wk2gt_4,373). happensAt(walking(wk2gt_4),373).
orientation(wk2gt_1,125,374). coords(wk2gt_1,115,53,374).  visible(wk2gt_1,374). happensAt(inactive(wk2gt_1),374). orientation(wk2gt_4,151,374). coords(wk2gt_4,165,134,374).  visible(wk2gt_4,374). happensAt(walking(wk2gt_4),374).
orientation(wk2gt_1,125,375). coords(wk2gt_1,115,53,375).  visible(wk2gt_1,375). happensAt(inactive(wk2gt_1),375). orientation(wk2gt_4,151,375). coords(wk2gt_4,164,133,375).  visible(wk2gt_4,375). happensAt(walking(wk2gt_4),375).
orientation(wk2gt_1,125,376). coords(wk2gt_1,115,53,376).  visible(wk2gt_1,376). happensAt(inactive(wk2gt_1),376). orientation(wk2gt_4,151,376). coords(wk2gt_4,163,133,376).  visible(wk2gt_4,376). happensAt(walking(wk2gt_4),376).
orientation(wk2gt_1,125,377). coords(wk2gt_1,115,53,377).  visible(wk2gt_1,377). happensAt(inactive(wk2gt_1),377). orientation(wk2gt_4,147,377). coords(wk2gt_4,161,131,377).  visible(wk2gt_4,377). happensAt(walking(wk2gt_4),377).
orientation(wk2gt_1,125,378). coords(wk2gt_1,115,53,378).  visible(wk2gt_1,378). happensAt(inactive(wk2gt_1),378). orientation(wk2gt_4,147,378). coords(wk2gt_4,159,130,378).  visible(wk2gt_4,378). happensAt(walking(wk2gt_4),378).
orientation(wk2gt_1,125,379). coords(wk2gt_1,115,53,379).  visible(wk2gt_1,379). happensAt(inactive(wk2gt_1),379). orientation(wk2gt_4,147,379). coords(wk2gt_4,156,128,379).  visible(wk2gt_4,379). happensAt(walking(wk2gt_4),379).
orientation(wk2gt_1,125,380). coords(wk2gt_1,115,53,380).  visible(wk2gt_1,380). happensAt(inactive(wk2gt_1),380). orientation(wk2gt_4,147,380). coords(wk2gt_4,155,126,380).  visible(wk2gt_4,380). happensAt(walking(wk2gt_4),380).
orientation(wk2gt_1,125,381). coords(wk2gt_1,115,53,381).  visible(wk2gt_1,381). happensAt(inactive(wk2gt_1),381). orientation(wk2gt_4,147,381). coords(wk2gt_4,153,125,381).  visible(wk2gt_4,381). happensAt(walking(wk2gt_4),381).
orientation(wk2gt_1,125,382). coords(wk2gt_1,115,53,382).  visible(wk2gt_1,382). happensAt(inactive(wk2gt_1),382). orientation(wk2gt_4,147,382). coords(wk2gt_4,153,124,382).  visible(wk2gt_4,382). happensAt(walking(wk2gt_4),382).
orientation(wk2gt_1,125,383). coords(wk2gt_1,115,53,383).  visible(wk2gt_1,383). happensAt(inactive(wk2gt_1),383). orientation(wk2gt_4,147,383). coords(wk2gt_4,152,122,383).  visible(wk2gt_4,383). happensAt(walking(wk2gt_4),383).
orientation(wk2gt_1,125,384). coords(wk2gt_1,115,53,384).  visible(wk2gt_1,384). happensAt(inactive(wk2gt_1),384). orientation(wk2gt_4,147,384). coords(wk2gt_4,152,122,384).  visible(wk2gt_4,384). happensAt(walking(wk2gt_4),384).
orientation(wk2gt_1,125,385). coords(wk2gt_1,115,53,385).  visible(wk2gt_1,385). happensAt(inactive(wk2gt_1),385). orientation(wk2gt_4,147,385). coords(wk2gt_4,151,121,385).  visible(wk2gt_4,385). happensAt(walking(wk2gt_4),385).
orientation(wk2gt_1,125,386). coords(wk2gt_1,115,53,386).  visible(wk2gt_1,386). happensAt(inactive(wk2gt_1),386). orientation(wk2gt_4,147,386). coords(wk2gt_4,150,121,386).  visible(wk2gt_4,386). happensAt(walking(wk2gt_4),386).
orientation(wk2gt_1,125,387). coords(wk2gt_1,115,53,387).  visible(wk2gt_1,387). happensAt(inactive(wk2gt_1),387). orientation(wk2gt_4,149,387). coords(wk2gt_4,150,121,387).  visible(wk2gt_4,387). happensAt(walking(wk2gt_4),387).
orientation(wk2gt_1,125,388). coords(wk2gt_1,115,53,388).  visible(wk2gt_1,388). happensAt(inactive(wk2gt_1),388). orientation(wk2gt_4,149,388). coords(wk2gt_4,149,120,388).  visible(wk2gt_4,388). happensAt(walking(wk2gt_4),388).
orientation(wk2gt_1,125,389). coords(wk2gt_1,115,53,389).  visible(wk2gt_1,389). happensAt(inactive(wk2gt_1),389). orientation(wk2gt_4,149,389). coords(wk2gt_4,148,119,389).  visible(wk2gt_4,389). happensAt(walking(wk2gt_4),389).
orientation(wk2gt_1,125,390). coords(wk2gt_1,115,53,390).  visible(wk2gt_1,390). happensAt(inactive(wk2gt_1),390). orientation(wk2gt_4,149,390). coords(wk2gt_4,146,118,390).  visible(wk2gt_4,390). happensAt(walking(wk2gt_4),390).
orientation(wk2gt_1,125,391). coords(wk2gt_1,115,53,391).  visible(wk2gt_1,391). happensAt(inactive(wk2gt_1),391). orientation(wk2gt_4,149,391). coords(wk2gt_4,144,117,391).  visible(wk2gt_4,391). happensAt(walking(wk2gt_4),391).
#show holdsAt/2.