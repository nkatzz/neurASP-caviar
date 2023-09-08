
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

	orientation(fcgt_1,150,376). coords(fcgt_1,83,100,376).  visible(fcgt_1,376). happensAt(running(fcgt_1),376). orientation(fcgt_2,142,376). coords(fcgt_2,84,94,376).  visible(fcgt_2,376). happensAt(running(fcgt_2),376).
orientation(fcgt_1,155,377). coords(fcgt_1,83,98,377).  visible(fcgt_1,377). happensAt(running(fcgt_1),377). orientation(fcgt_2,142,377). coords(fcgt_2,84,93,377).  visible(fcgt_2,377). happensAt(running(fcgt_2),377).
orientation(fcgt_1,154,378). coords(fcgt_1,82,98,378).  visible(fcgt_1,378). happensAt(running(fcgt_1),378). orientation(fcgt_2,142,378). coords(fcgt_2,84,93,378).  visible(fcgt_2,378). happensAt(running(fcgt_2),378).
orientation(fcgt_1,155,379). coords(fcgt_1,82,97,379).  visible(fcgt_1,379). happensAt(running(fcgt_1),379). orientation(fcgt_2,142,379). coords(fcgt_2,84,93,379).  visible(fcgt_2,379). happensAt(running(fcgt_2),379).
orientation(fcgt_1,157,380). coords(fcgt_1,82,97,380).  visible(fcgt_1,380). happensAt(running(fcgt_1),380). orientation(fcgt_2,142,380). coords(fcgt_2,84,93,380).  visible(fcgt_2,380). happensAt(running(fcgt_2),380).
orientation(fcgt_1,157,381). coords(fcgt_1,82,96,381).  visible(fcgt_1,381). happensAt(running(fcgt_1),381). orientation(fcgt_2,142,381). coords(fcgt_2,84,93,381).  visible(fcgt_2,381). happensAt(running(fcgt_2),381).
orientation(fcgt_1,157,382). coords(fcgt_1,81,96,382).  visible(fcgt_1,382). happensAt(running(fcgt_1),382). orientation(fcgt_2,142,382). coords(fcgt_2,83,92,382).  visible(fcgt_2,382). happensAt(running(fcgt_2),382).
orientation(fcgt_1,157,383). coords(fcgt_1,80,96,383).  visible(fcgt_1,383). happensAt(running(fcgt_1),383). orientation(fcgt_2,142,383). coords(fcgt_2,83,92,383).  visible(fcgt_2,383). happensAt(running(fcgt_2),383).
orientation(fcgt_1,157,384). coords(fcgt_1,79,96,384).  visible(fcgt_1,384). happensAt(running(fcgt_1),384). orientation(fcgt_2,142,384). coords(fcgt_2,82,91,384).  visible(fcgt_2,384). happensAt(running(fcgt_2),384).
orientation(fcgt_1,157,385). coords(fcgt_1,78,96,385).  visible(fcgt_1,385). happensAt(running(fcgt_1),385). orientation(fcgt_2,142,385). coords(fcgt_2,81,90,385).  visible(fcgt_2,385). happensAt(running(fcgt_2),385).
orientation(fcgt_1,152,386). coords(fcgt_1,78,96,386).  visible(fcgt_1,386). happensAt(running(fcgt_1),386). orientation(fcgt_2,142,386). coords(fcgt_2,81,89,386).  visible(fcgt_2,386). happensAt(running(fcgt_2),386).
orientation(fcgt_1,146,387). coords(fcgt_1,77,96,387).  visible(fcgt_1,387). happensAt(running(fcgt_1),387). orientation(fcgt_2,142,387). coords(fcgt_2,80,89,387).  visible(fcgt_2,387). happensAt(running(fcgt_2),387).
orientation(fcgt_1,146,388). coords(fcgt_1,77,96,388).  visible(fcgt_1,388). happensAt(running(fcgt_1),388). orientation(fcgt_2,142,388). coords(fcgt_2,79,88,388).  visible(fcgt_2,388). happensAt(running(fcgt_2),388).
orientation(fcgt_1,140,389). coords(fcgt_1,76,95,389).  visible(fcgt_1,389). happensAt(running(fcgt_1),389). orientation(fcgt_2,142,389). coords(fcgt_2,79,87,389).  visible(fcgt_2,389). happensAt(running(fcgt_2),389).
orientation(fcgt_1,137,390). coords(fcgt_1,76,95,390).  visible(fcgt_1,390). happensAt(running(fcgt_1),390). orientation(fcgt_2,139,390). coords(fcgt_2,79,87,390).  visible(fcgt_2,390). happensAt(running(fcgt_2),390).
orientation(fcgt_1,137,391). coords(fcgt_1,75,93,391).  visible(fcgt_1,391). happensAt(running(fcgt_1),391). orientation(fcgt_2,137,391). coords(fcgt_2,79,86,391).  visible(fcgt_2,391). happensAt(running(fcgt_2),391).
orientation(fcgt_1,137,392). coords(fcgt_1,75,93,392).  visible(fcgt_1,392). happensAt(running(fcgt_1),392). orientation(fcgt_2,134,392). coords(fcgt_2,78,86,392).  visible(fcgt_2,392). happensAt(running(fcgt_2),392).
orientation(fcgt_1,137,393). coords(fcgt_1,75,91,393).  visible(fcgt_1,393). happensAt(running(fcgt_1),393). orientation(fcgt_2,130,393). coords(fcgt_2,78,85,393).  visible(fcgt_2,393). happensAt(running(fcgt_2),393).
orientation(fcgt_1,137,394). coords(fcgt_1,75,90,394).  visible(fcgt_1,394). happensAt(running(fcgt_1),394). orientation(fcgt_2,130,394). coords(fcgt_2,78,84,394).  visible(fcgt_2,394). happensAt(running(fcgt_2),394).
orientation(fcgt_1,137,395). coords(fcgt_1,75,90,395).  visible(fcgt_1,395). happensAt(running(fcgt_1),395). orientation(fcgt_2,130,395). coords(fcgt_2,78,82,395).  visible(fcgt_2,395). happensAt(running(fcgt_2),395).
orientation(fcgt_1,137,396). coords(fcgt_1,75,88,396).  visible(fcgt_1,396). happensAt(running(fcgt_1),396). orientation(fcgt_2,130,396). coords(fcgt_2,78,81,396).  visible(fcgt_2,396). happensAt(running(fcgt_2),396).
orientation(fcgt_1,137,397). coords(fcgt_1,73,87,397).  visible(fcgt_1,397). happensAt(running(fcgt_1),397). orientation(fcgt_2,130,397). coords(fcgt_2,78,81,397).  visible(fcgt_2,397). happensAt(running(fcgt_2),397).
orientation(fcgt_1,137,398). coords(fcgt_1,73,86,398).  visible(fcgt_1,398). happensAt(running(fcgt_1),398). orientation(fcgt_2,130,398). coords(fcgt_2,77,79,398).  visible(fcgt_2,398). happensAt(running(fcgt_2),398).
orientation(fcgt_1,137,399). coords(fcgt_1,72,85,399).  visible(fcgt_1,399). happensAt(running(fcgt_1),399). orientation(fcgt_2,130,399). coords(fcgt_2,77,78,399).  visible(fcgt_2,399). happensAt(running(fcgt_2),399).
orientation(fcgt_1,137,400). coords(fcgt_1,72,85,400).  visible(fcgt_1,400). happensAt(running(fcgt_1),400). orientation(fcgt_2,130,400). coords(fcgt_2,77,77,400).  visible(fcgt_2,400). happensAt(running(fcgt_2),400).
#show holdsAt/2.