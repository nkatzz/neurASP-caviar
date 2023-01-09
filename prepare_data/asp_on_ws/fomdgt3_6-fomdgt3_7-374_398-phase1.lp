
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

	orientation(fomdgt3_6,156,374). coords(fomdgt3_6,130,92,374).  visible(fomdgt3_6,374). happensAt(running(fomdgt3_6),374). orientation(fomdgt3_7,134,374). coords(fomdgt3_7,78,85,374).  visible(fomdgt3_7,374). happensAt(walking(fomdgt3_7),374).
orientation(fomdgt3_6,156,375). coords(fomdgt3_6,127,91,375).  visible(fomdgt3_6,375). happensAt(running(fomdgt3_6),375). orientation(fomdgt3_7,134,375). coords(fomdgt3_7,78,86,375).  visible(fomdgt3_7,375). happensAt(walking(fomdgt3_7),375).
orientation(fomdgt3_6,156,376). coords(fomdgt3_6,124,91,376).  visible(fomdgt3_6,376). happensAt(running(fomdgt3_6),376). orientation(fomdgt3_7,134,376). coords(fomdgt3_7,77,87,376).  visible(fomdgt3_7,376). happensAt(walking(fomdgt3_7),376).
orientation(fomdgt3_6,156,377). coords(fomdgt3_6,120,91,377).  visible(fomdgt3_6,377). happensAt(running(fomdgt3_6),377). orientation(fomdgt3_7,134,377). coords(fomdgt3_7,77,88,377).  visible(fomdgt3_7,377). happensAt(walking(fomdgt3_7),377).
orientation(fomdgt3_6,156,378). coords(fomdgt3_6,118,91,378).  visible(fomdgt3_6,378). happensAt(running(fomdgt3_6),378). orientation(fomdgt3_7,134,378). coords(fomdgt3_7,77,89,378).  visible(fomdgt3_7,378). happensAt(walking(fomdgt3_7),378).
orientation(fomdgt3_6,156,379). coords(fomdgt3_6,116,91,379).  visible(fomdgt3_6,379). happensAt(running(fomdgt3_6),379). orientation(fomdgt3_7,134,379). coords(fomdgt3_7,77,90,379).  visible(fomdgt3_7,379). happensAt(walking(fomdgt3_7),379).
orientation(fomdgt3_6,156,380). coords(fomdgt3_6,115,91,380).  visible(fomdgt3_6,380). happensAt(running(fomdgt3_6),380). orientation(fomdgt3_7,134,380). coords(fomdgt3_7,77,91,380).  visible(fomdgt3_7,380). happensAt(walking(fomdgt3_7),380).
orientation(fomdgt3_6,156,381). coords(fomdgt3_6,114,90,381).  visible(fomdgt3_6,381). happensAt(running(fomdgt3_6),381). orientation(fomdgt3_7,134,381). coords(fomdgt3_7,77,91,381).  visible(fomdgt3_7,381). happensAt(walking(fomdgt3_7),381).
orientation(fomdgt3_6,156,382). coords(fomdgt3_6,113,90,382).  visible(fomdgt3_6,382). happensAt(running(fomdgt3_6),382). orientation(fomdgt3_7,134,382). coords(fomdgt3_7,77,91,382).  visible(fomdgt3_7,382). happensAt(walking(fomdgt3_7),382).
orientation(fomdgt3_6,156,383). coords(fomdgt3_6,108,89,383).  visible(fomdgt3_6,383). happensAt(running(fomdgt3_6),383). orientation(fomdgt3_7,134,383). coords(fomdgt3_7,77,91,383).  visible(fomdgt3_7,383). happensAt(walking(fomdgt3_7),383).
orientation(fomdgt3_6,140,384). coords(fomdgt3_6,106,89,384).  visible(fomdgt3_6,384). happensAt(running(fomdgt3_6),384). orientation(fomdgt3_7,134,384). coords(fomdgt3_7,77,92,384).  visible(fomdgt3_7,384). happensAt(walking(fomdgt3_7),384).
orientation(fomdgt3_6,140,385). coords(fomdgt3_6,103,88,385).  visible(fomdgt3_6,385). happensAt(running(fomdgt3_6),385). orientation(fomdgt3_7,134,385). coords(fomdgt3_7,77,93,385).  visible(fomdgt3_7,385). happensAt(walking(fomdgt3_7),385).
orientation(fomdgt3_6,140,386). coords(fomdgt3_6,102,88,386).  visible(fomdgt3_6,386). happensAt(running(fomdgt3_6),386). orientation(fomdgt3_7,134,386). coords(fomdgt3_7,77,93,386).  visible(fomdgt3_7,386). happensAt(walking(fomdgt3_7),386).
orientation(fomdgt3_6,140,387). coords(fomdgt3_6,100,88,387).  visible(fomdgt3_6,387). happensAt(running(fomdgt3_6),387). orientation(fomdgt3_7,134,387). coords(fomdgt3_7,77,93,387).  visible(fomdgt3_7,387). happensAt(walking(fomdgt3_7),387).
orientation(fomdgt3_6,140,388). coords(fomdgt3_6,100,88,388).  visible(fomdgt3_6,388). happensAt(running(fomdgt3_6),388). orientation(fomdgt3_7,134,388). coords(fomdgt3_7,77,93,388).  visible(fomdgt3_7,388). happensAt(walking(fomdgt3_7),388).
orientation(fomdgt3_6,140,389). coords(fomdgt3_6,99,88,389).  visible(fomdgt3_6,389). happensAt(running(fomdgt3_6),389). orientation(fomdgt3_7,134,389). coords(fomdgt3_7,77,93,389).  visible(fomdgt3_7,389). happensAt(walking(fomdgt3_7),389).
orientation(fomdgt3_6,140,390). coords(fomdgt3_6,96,85,390).  visible(fomdgt3_6,390). happensAt(running(fomdgt3_6),390). orientation(fomdgt3_7,134,390). coords(fomdgt3_7,77,93,390).  visible(fomdgt3_7,390). happensAt(walking(fomdgt3_7),390).
orientation(fomdgt3_6,140,391). coords(fomdgt3_6,93,84,391).  visible(fomdgt3_6,391). happensAt(running(fomdgt3_6),391). orientation(fomdgt3_7,134,391). coords(fomdgt3_7,77,94,391).  visible(fomdgt3_7,391). happensAt(walking(fomdgt3_7),391).
orientation(fomdgt3_6,140,392). coords(fomdgt3_6,92,83,392).  visible(fomdgt3_6,392). happensAt(running(fomdgt3_6),392). orientation(fomdgt3_7,134,392). coords(fomdgt3_7,78,96,392).  visible(fomdgt3_7,392). happensAt(walking(fomdgt3_7),392).
orientation(fomdgt3_6,140,393). coords(fomdgt3_6,92,83,393).  visible(fomdgt3_6,393). happensAt(running(fomdgt3_6),393). orientation(fomdgt3_7,134,393). coords(fomdgt3_7,78,97,393).  visible(fomdgt3_7,393). happensAt(walking(fomdgt3_7),393).
orientation(fomdgt3_6,140,394). coords(fomdgt3_6,91,83,394).  visible(fomdgt3_6,394). happensAt(running(fomdgt3_6),394). orientation(fomdgt3_7,134,394). coords(fomdgt3_7,78,97,394).  visible(fomdgt3_7,394). happensAt(walking(fomdgt3_7),394).
orientation(fomdgt3_6,140,395). coords(fomdgt3_6,90,82,395).  visible(fomdgt3_6,395). happensAt(running(fomdgt3_6),395). orientation(fomdgt3_7,134,395). coords(fomdgt3_7,79,97,395).  visible(fomdgt3_7,395). happensAt(walking(fomdgt3_7),395).
orientation(fomdgt3_6,140,396). coords(fomdgt3_6,88,82,396).  visible(fomdgt3_6,396). happensAt(running(fomdgt3_6),396). orientation(fomdgt3_7,134,396). coords(fomdgt3_7,79,98,396).  visible(fomdgt3_7,396). happensAt(walking(fomdgt3_7),396).
orientation(fomdgt3_6,140,397). coords(fomdgt3_6,87,81,397).  visible(fomdgt3_6,397). happensAt(running(fomdgt3_6),397). orientation(fomdgt3_7,134,397). coords(fomdgt3_7,79,98,397).  visible(fomdgt3_7,397). happensAt(walking(fomdgt3_7),397).
orientation(fomdgt3_6,140,398). coords(fomdgt3_6,85,81,398).  visible(fomdgt3_6,398). happensAt(running(fomdgt3_6),398). orientation(fomdgt3_7,134,398). coords(fomdgt3_7,79,99,398).  visible(fomdgt3_7,398). happensAt(walking(fomdgt3_7),398).
#show holdsAt/2.