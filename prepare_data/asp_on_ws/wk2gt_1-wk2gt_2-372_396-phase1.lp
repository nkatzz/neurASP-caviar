
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

	orientation(wk2gt_1,125,372). coords(wk2gt_1,115,53,372).  visible(wk2gt_1,372). happensAt(inactive(wk2gt_1),372). orientation(wk2gt_2,141,372). coords(wk2gt_2,98,63,372).  visible(wk2gt_2,372). happensAt(inactive(wk2gt_2),372).
orientation(wk2gt_1,125,373). coords(wk2gt_1,115,53,373).  visible(wk2gt_1,373). happensAt(inactive(wk2gt_1),373). orientation(wk2gt_2,141,373). coords(wk2gt_2,98,63,373).  visible(wk2gt_2,373). happensAt(inactive(wk2gt_2),373).
orientation(wk2gt_1,125,374). coords(wk2gt_1,115,53,374).  visible(wk2gt_1,374). happensAt(inactive(wk2gt_1),374). orientation(wk2gt_2,141,374). coords(wk2gt_2,98,63,374).  visible(wk2gt_2,374). happensAt(inactive(wk2gt_2),374).
orientation(wk2gt_1,125,375). coords(wk2gt_1,115,53,375).  visible(wk2gt_1,375). happensAt(inactive(wk2gt_1),375). orientation(wk2gt_2,141,375). coords(wk2gt_2,98,63,375).  visible(wk2gt_2,375). happensAt(inactive(wk2gt_2),375).
orientation(wk2gt_1,125,376). coords(wk2gt_1,115,53,376).  visible(wk2gt_1,376). happensAt(inactive(wk2gt_1),376). orientation(wk2gt_2,141,376). coords(wk2gt_2,98,63,376).  visible(wk2gt_2,376). happensAt(inactive(wk2gt_2),376).
orientation(wk2gt_1,125,377). coords(wk2gt_1,115,53,377).  visible(wk2gt_1,377). happensAt(inactive(wk2gt_1),377). orientation(wk2gt_2,141,377). coords(wk2gt_2,98,63,377).  visible(wk2gt_2,377). happensAt(inactive(wk2gt_2),377).
orientation(wk2gt_1,125,378). coords(wk2gt_1,115,53,378).  visible(wk2gt_1,378). happensAt(inactive(wk2gt_1),378). orientation(wk2gt_2,141,378). coords(wk2gt_2,98,63,378).  visible(wk2gt_2,378). happensAt(inactive(wk2gt_2),378).
orientation(wk2gt_1,125,379). coords(wk2gt_1,115,53,379).  visible(wk2gt_1,379). happensAt(inactive(wk2gt_1),379). orientation(wk2gt_2,141,379). coords(wk2gt_2,98,63,379).  visible(wk2gt_2,379). happensAt(inactive(wk2gt_2),379).
orientation(wk2gt_1,125,380). coords(wk2gt_1,115,53,380).  visible(wk2gt_1,380). happensAt(inactive(wk2gt_1),380). orientation(wk2gt_2,141,380). coords(wk2gt_2,98,63,380).  visible(wk2gt_2,380). happensAt(inactive(wk2gt_2),380).
orientation(wk2gt_1,125,381). coords(wk2gt_1,115,53,381).  visible(wk2gt_1,381). happensAt(inactive(wk2gt_1),381). orientation(wk2gt_2,141,381). coords(wk2gt_2,98,63,381).  visible(wk2gt_2,381). happensAt(inactive(wk2gt_2),381).
orientation(wk2gt_1,125,382). coords(wk2gt_1,115,53,382).  visible(wk2gt_1,382). happensAt(inactive(wk2gt_1),382). orientation(wk2gt_2,141,382). coords(wk2gt_2,98,63,382).  visible(wk2gt_2,382). happensAt(inactive(wk2gt_2),382).
orientation(wk2gt_1,125,383). coords(wk2gt_1,115,53,383).  visible(wk2gt_1,383). happensAt(inactive(wk2gt_1),383). orientation(wk2gt_2,141,383). coords(wk2gt_2,98,63,383).  visible(wk2gt_2,383). happensAt(inactive(wk2gt_2),383).
orientation(wk2gt_1,125,384). coords(wk2gt_1,115,53,384).  visible(wk2gt_1,384). happensAt(inactive(wk2gt_1),384). orientation(wk2gt_2,141,384). coords(wk2gt_2,98,63,384).  visible(wk2gt_2,384). happensAt(inactive(wk2gt_2),384).
orientation(wk2gt_1,125,385). coords(wk2gt_1,115,53,385).  visible(wk2gt_1,385). happensAt(inactive(wk2gt_1),385). orientation(wk2gt_2,141,385). coords(wk2gt_2,98,63,385).  visible(wk2gt_2,385). happensAt(inactive(wk2gt_2),385).
orientation(wk2gt_1,125,386). coords(wk2gt_1,115,53,386).  visible(wk2gt_1,386). happensAt(inactive(wk2gt_1),386). orientation(wk2gt_2,141,386). coords(wk2gt_2,98,63,386).  visible(wk2gt_2,386). happensAt(inactive(wk2gt_2),386).
orientation(wk2gt_1,125,387). coords(wk2gt_1,115,53,387).  visible(wk2gt_1,387). happensAt(inactive(wk2gt_1),387). orientation(wk2gt_2,141,387). coords(wk2gt_2,98,63,387).  visible(wk2gt_2,387). happensAt(inactive(wk2gt_2),387).
orientation(wk2gt_1,125,388). coords(wk2gt_1,115,53,388).  visible(wk2gt_1,388). happensAt(inactive(wk2gt_1),388). orientation(wk2gt_2,141,388). coords(wk2gt_2,98,63,388).  visible(wk2gt_2,388). happensAt(inactive(wk2gt_2),388).
orientation(wk2gt_1,125,389). coords(wk2gt_1,115,53,389).  visible(wk2gt_1,389). happensAt(inactive(wk2gt_1),389). orientation(wk2gt_2,141,389). coords(wk2gt_2,98,63,389).  visible(wk2gt_2,389). happensAt(inactive(wk2gt_2),389).
orientation(wk2gt_1,125,390). coords(wk2gt_1,115,53,390).  visible(wk2gt_1,390). happensAt(inactive(wk2gt_1),390). orientation(wk2gt_2,141,390). coords(wk2gt_2,98,63,390).  visible(wk2gt_2,390). happensAt(inactive(wk2gt_2),390).
orientation(wk2gt_1,125,391). coords(wk2gt_1,115,53,391).  visible(wk2gt_1,391). happensAt(inactive(wk2gt_1),391). orientation(wk2gt_2,141,391). coords(wk2gt_2,98,63,391).  visible(wk2gt_2,391). happensAt(inactive(wk2gt_2),391).
orientation(wk2gt_1,125,392). coords(wk2gt_1,115,53,392).  visible(wk2gt_1,392). happensAt(inactive(wk2gt_1),392). orientation(wk2gt_2,141,392). coords(wk2gt_2,98,63,392).  visible(wk2gt_2,392). happensAt(inactive(wk2gt_2),392).
orientation(wk2gt_1,125,393). coords(wk2gt_1,115,53,393).  visible(wk2gt_1,393). happensAt(inactive(wk2gt_1),393). orientation(wk2gt_2,141,393). coords(wk2gt_2,98,63,393).  visible(wk2gt_2,393). happensAt(inactive(wk2gt_2),393).
orientation(wk2gt_1,125,394). coords(wk2gt_1,115,53,394).  visible(wk2gt_1,394). happensAt(inactive(wk2gt_1),394). orientation(wk2gt_2,141,394). coords(wk2gt_2,98,63,394).  visible(wk2gt_2,394). happensAt(inactive(wk2gt_2),394).
orientation(wk2gt_1,125,395). coords(wk2gt_1,115,53,395).  visible(wk2gt_1,395). happensAt(inactive(wk2gt_1),395). orientation(wk2gt_2,141,395). coords(wk2gt_2,98,63,395).  visible(wk2gt_2,395). happensAt(inactive(wk2gt_2),395).
orientation(wk2gt_1,125,396). coords(wk2gt_1,115,53,396).  visible(wk2gt_1,396). happensAt(inactive(wk2gt_1),396). orientation(wk2gt_2,141,396). coords(wk2gt_2,98,63,396).  visible(wk2gt_2,396). happensAt(inactive(wk2gt_2),396).
#show holdsAt/2.