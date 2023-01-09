
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

	orientation(bww1gt_1,0,374). coords(bww1gt_1,105,264,374).  visible(bww1gt_1,374). happensAt(inactive(bww1gt_1),374). orientation(bww1gt_2,160,374). coords(bww1gt_2,95,152,374).  visible(bww1gt_2,374). happensAt(walking(bww1gt_2),374).
orientation(bww1gt_1,1,375). coords(bww1gt_1,105,264,375).  visible(bww1gt_1,375). happensAt(inactive(bww1gt_1),375). orientation(bww1gt_2,162,375). coords(bww1gt_2,96,152,375).  visible(bww1gt_2,375). happensAt(walking(bww1gt_2),375).
orientation(bww1gt_1,3,376). coords(bww1gt_1,105,264,376).  visible(bww1gt_1,376). happensAt(inactive(bww1gt_1),376). orientation(bww1gt_2,161,376). coords(bww1gt_2,96,152,376).  visible(bww1gt_2,376). happensAt(walking(bww1gt_2),376).
orientation(bww1gt_1,4,377). coords(bww1gt_1,105,264,377).  visible(bww1gt_1,377). happensAt(inactive(bww1gt_1),377). orientation(bww1gt_2,159,377). coords(bww1gt_2,96,151,377).  visible(bww1gt_2,377). happensAt(walking(bww1gt_2),377).
orientation(bww1gt_1,2,378). coords(bww1gt_1,105,264,378).  visible(bww1gt_1,378). happensAt(inactive(bww1gt_1),378). orientation(bww1gt_2,157,378). coords(bww1gt_2,96,151,378).  visible(bww1gt_2,378). happensAt(walking(bww1gt_2),378).
orientation(bww1gt_1,3,379). coords(bww1gt_1,105,264,379).  visible(bww1gt_1,379). happensAt(inactive(bww1gt_1),379). orientation(bww1gt_2,158,379). coords(bww1gt_2,96,151,379).  visible(bww1gt_2,379). happensAt(walking(bww1gt_2),379).
orientation(bww1gt_1,5,380). coords(bww1gt_1,105,264,380).  visible(bww1gt_1,380). happensAt(inactive(bww1gt_1),380). orientation(bww1gt_2,159,380). coords(bww1gt_2,97,151,380).  visible(bww1gt_2,380). happensAt(walking(bww1gt_2),380).
orientation(bww1gt_1,4,381). coords(bww1gt_1,105,264,381).  visible(bww1gt_1,381). happensAt(inactive(bww1gt_1),381). orientation(bww1gt_2,157,381). coords(bww1gt_2,97,151,381).  visible(bww1gt_2,381). happensAt(walking(bww1gt_2),381).
orientation(bww1gt_1,1,382). coords(bww1gt_1,105,264,382).  visible(bww1gt_1,382). happensAt(inactive(bww1gt_1),382). orientation(bww1gt_2,158,382). coords(bww1gt_2,98,150,382).  visible(bww1gt_2,382). happensAt(walking(bww1gt_2),382).
orientation(bww1gt_1,178,383). coords(bww1gt_1,105,264,383).  visible(bww1gt_1,383). happensAt(inactive(bww1gt_1),383). orientation(bww1gt_2,163,383). coords(bww1gt_2,101,150,383).  visible(bww1gt_2,383). happensAt(walking(bww1gt_2),383).
orientation(bww1gt_1,179,384). coords(bww1gt_1,105,264,384).  visible(bww1gt_1,384). happensAt(inactive(bww1gt_1),384). orientation(bww1gt_2,160,384). coords(bww1gt_2,102,149,384).  visible(bww1gt_2,384). happensAt(walking(bww1gt_2),384).
orientation(bww1gt_1,178,385). coords(bww1gt_1,105,264,385).  visible(bww1gt_1,385). happensAt(inactive(bww1gt_1),385). orientation(bww1gt_2,157,385). coords(bww1gt_2,103,149,385).  visible(bww1gt_2,385). happensAt(walking(bww1gt_2),385).
orientation(bww1gt_1,179,386). coords(bww1gt_1,105,264,386).  visible(bww1gt_1,386). happensAt(inactive(bww1gt_1),386). orientation(bww1gt_2,158,386). coords(bww1gt_2,104,149,386).  visible(bww1gt_2,386). happensAt(walking(bww1gt_2),386).
orientation(bww1gt_1,179,387). coords(bww1gt_1,105,264,387).  visible(bww1gt_1,387). happensAt(inactive(bww1gt_1),387). orientation(bww1gt_2,157,387). coords(bww1gt_2,105,149,387).  visible(bww1gt_2,387). happensAt(walking(bww1gt_2),387).
orientation(bww1gt_1,2,388). coords(bww1gt_1,105,264,388).  visible(bww1gt_1,388). happensAt(inactive(bww1gt_1),388). orientation(bww1gt_2,160,388). coords(bww1gt_2,106,149,388).  visible(bww1gt_2,388). happensAt(walking(bww1gt_2),388).
orientation(bww1gt_1,3,389). coords(bww1gt_1,105,264,389).  visible(bww1gt_1,389). happensAt(inactive(bww1gt_1),389). orientation(bww1gt_2,157,389). coords(bww1gt_2,106,148,389).  visible(bww1gt_2,389). happensAt(walking(bww1gt_2),389).
orientation(bww1gt_1,1,390). coords(bww1gt_1,105,264,390).  visible(bww1gt_1,390). happensAt(inactive(bww1gt_1),390). orientation(bww1gt_2,154,390). coords(bww1gt_2,106,148,390).  visible(bww1gt_2,390). happensAt(walking(bww1gt_2),390).
orientation(bww1gt_1,2,391). coords(bww1gt_1,105,264,391).  visible(bww1gt_1,391). happensAt(inactive(bww1gt_1),391). orientation(bww1gt_2,158,391). coords(bww1gt_2,107,147,391).  visible(bww1gt_2,391). happensAt(walking(bww1gt_2),391).
orientation(bww1gt_1,0,392). coords(bww1gt_1,105,264,392).  visible(bww1gt_1,392). happensAt(inactive(bww1gt_1),392). orientation(bww1gt_2,157,392). coords(bww1gt_2,108,147,392).  visible(bww1gt_2,392). happensAt(walking(bww1gt_2),392).
orientation(bww1gt_1,3,393). coords(bww1gt_1,105,264,393).  visible(bww1gt_1,393). happensAt(inactive(bww1gt_1),393). orientation(bww1gt_2,160,393). coords(bww1gt_2,108,147,393).  visible(bww1gt_2,393). happensAt(walking(bww1gt_2),393).
orientation(bww1gt_1,7,394). coords(bww1gt_1,105,264,394).  visible(bww1gt_1,394). happensAt(inactive(bww1gt_1),394). orientation(bww1gt_2,156,394). coords(bww1gt_2,108,146,394).  visible(bww1gt_2,394). happensAt(walking(bww1gt_2),394).
orientation(bww1gt_1,1,395). coords(bww1gt_1,105,264,395).  visible(bww1gt_1,395). happensAt(inactive(bww1gt_1),395). orientation(bww1gt_2,158,395). coords(bww1gt_2,109,146,395).  visible(bww1gt_2,395). happensAt(walking(bww1gt_2),395).
orientation(bww1gt_1,2,396). coords(bww1gt_1,105,264,396).  visible(bww1gt_1,396). happensAt(inactive(bww1gt_1),396). orientation(bww1gt_2,157,396). coords(bww1gt_2,109,146,396).  visible(bww1gt_2,396). happensAt(walking(bww1gt_2),396).
orientation(bww1gt_1,3,397). coords(bww1gt_1,105,264,397).  visible(bww1gt_1,397). happensAt(inactive(bww1gt_1),397). orientation(bww1gt_2,156,397). coords(bww1gt_2,109,146,397).  visible(bww1gt_2,397). happensAt(walking(bww1gt_2),397).
orientation(bww1gt_1,179,398). coords(bww1gt_1,105,264,398).  visible(bww1gt_1,398). happensAt(inactive(bww1gt_1),398). orientation(bww1gt_2,155,398). coords(bww1gt_2,110,146,398).  visible(bww1gt_2,398). happensAt(walking(bww1gt_2),398).
#show holdsAt/2.