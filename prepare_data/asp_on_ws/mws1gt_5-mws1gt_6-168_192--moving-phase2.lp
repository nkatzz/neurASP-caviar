

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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).


happensAt(active(X),T) :- simpleEvent(T,X,active).
happensAt(inactive(X),T) :- simpleEvent(T,X,inactive).
happensAt(walking(X),T) :- simpleEvent(T,X,walking).
happensAt(running(X),T) :- simpleEvent(T,X,running). 


holdsByInertia(moving(X,Y),T+1) :- holdsAt(moving(X,Y),T+1), not initiatedAt(moving(X,Y),T).
holdsByInertia(meeting(X,Y),T+1) :- holdsAt(meeting(X,Y),T+1), not initiatedAt(meeting(X,Y),T).

:- holdsByInertia(moving(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2. 
:- holdsByInertia(meeting(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2.

	person(mws1gt_5;mws1gt_6).
orientation(mws1gt_5,145,168). coords(mws1gt_5,64,89,168).  visible(mws1gt_5,168). orientation(mws1gt_6,146,168). coords(mws1gt_6,73,65,168).  visible(mws1gt_6,168).
orientation(mws1gt_5,145,169). coords(mws1gt_5,64,90,169).  visible(mws1gt_5,169). orientation(mws1gt_6,146,169). coords(mws1gt_6,73,66,169).  visible(mws1gt_6,169).
orientation(mws1gt_5,145,170). coords(mws1gt_5,64,90,170).  visible(mws1gt_5,170). orientation(mws1gt_6,146,170). coords(mws1gt_6,73,67,170).  visible(mws1gt_6,170).
orientation(mws1gt_5,145,171). coords(mws1gt_5,63,90,171).  visible(mws1gt_5,171). orientation(mws1gt_6,146,171). coords(mws1gt_6,72,67,171).  visible(mws1gt_6,171).
orientation(mws1gt_5,145,172). coords(mws1gt_5,63,90,172).  visible(mws1gt_5,172). orientation(mws1gt_6,146,172). coords(mws1gt_6,72,67,172).  visible(mws1gt_6,172).
orientation(mws1gt_5,145,173). coords(mws1gt_5,62,90,173).  visible(mws1gt_5,173). orientation(mws1gt_6,146,173). coords(mws1gt_6,72,68,173).  visible(mws1gt_6,173).
orientation(mws1gt_5,145,174). coords(mws1gt_5,62,90,174).  visible(mws1gt_5,174). orientation(mws1gt_6,140,174). coords(mws1gt_6,72,68,174).  visible(mws1gt_6,174).
orientation(mws1gt_5,145,175). coords(mws1gt_5,61,90,175).  visible(mws1gt_5,175). orientation(mws1gt_6,140,175). coords(mws1gt_6,71,71,175).  visible(mws1gt_6,175).
orientation(mws1gt_5,145,176). coords(mws1gt_5,61,91,176).  visible(mws1gt_5,176). orientation(mws1gt_6,140,176). coords(mws1gt_6,70,71,176).  visible(mws1gt_6,176).
orientation(mws1gt_5,145,177). coords(mws1gt_5,60,91,177).  visible(mws1gt_5,177). orientation(mws1gt_6,140,177). coords(mws1gt_6,70,72,177).  visible(mws1gt_6,177).
orientation(mws1gt_5,145,178). coords(mws1gt_5,59,93,178).  visible(mws1gt_5,178). orientation(mws1gt_6,140,178). coords(mws1gt_6,70,73,178).  visible(mws1gt_6,178).
orientation(mws1gt_5,145,179). coords(mws1gt_5,59,94,179).  visible(mws1gt_5,179). orientation(mws1gt_6,140,179). coords(mws1gt_6,70,73,179).  visible(mws1gt_6,179).
orientation(mws1gt_5,145,180). coords(mws1gt_5,58,95,180).  visible(mws1gt_5,180). orientation(mws1gt_6,140,180). coords(mws1gt_6,70,73,180).  visible(mws1gt_6,180).
orientation(mws1gt_5,145,181). coords(mws1gt_5,58,96,181).  visible(mws1gt_5,181). orientation(mws1gt_6,140,181). coords(mws1gt_6,69,74,181).  visible(mws1gt_6,181).
orientation(mws1gt_5,145,182). coords(mws1gt_5,58,96,182).  visible(mws1gt_5,182). orientation(mws1gt_6,140,182). coords(mws1gt_6,69,74,182).  visible(mws1gt_6,182).
orientation(mws1gt_5,145,183). coords(mws1gt_5,58,97,183).  visible(mws1gt_5,183). orientation(mws1gt_6,140,183). coords(mws1gt_6,69,75,183).  visible(mws1gt_6,183).
orientation(mws1gt_5,145,184). coords(mws1gt_5,58,97,184).  visible(mws1gt_5,184). orientation(mws1gt_6,140,184). coords(mws1gt_6,68,76,184).  visible(mws1gt_6,184).
orientation(mws1gt_5,145,185). coords(mws1gt_5,57,97,185).  visible(mws1gt_5,185). orientation(mws1gt_6,140,185). coords(mws1gt_6,68,76,185).  visible(mws1gt_6,185).
orientation(mws1gt_5,145,186). coords(mws1gt_5,57,97,186).  visible(mws1gt_5,186). orientation(mws1gt_6,140,186). coords(mws1gt_6,68,76,186).  visible(mws1gt_6,186).
orientation(mws1gt_5,145,187). coords(mws1gt_5,57,98,187).  visible(mws1gt_5,187). orientation(mws1gt_6,140,187). coords(mws1gt_6,68,77,187).  visible(mws1gt_6,187).
orientation(mws1gt_5,145,188). coords(mws1gt_5,56,98,188).  visible(mws1gt_5,188). orientation(mws1gt_6,140,188). coords(mws1gt_6,67,77,188).  visible(mws1gt_6,188).
orientation(mws1gt_5,145,189). coords(mws1gt_5,55,99,189).  visible(mws1gt_5,189). orientation(mws1gt_6,140,189). coords(mws1gt_6,66,78,189).  visible(mws1gt_6,189).
orientation(mws1gt_5,145,190). coords(mws1gt_5,55,99,190).  visible(mws1gt_5,190). orientation(mws1gt_6,140,190). coords(mws1gt_6,66,79,190).  visible(mws1gt_6,190).
orientation(mws1gt_5,145,191). coords(mws1gt_5,54,99,191).  visible(mws1gt_5,191). orientation(mws1gt_6,140,191). coords(mws1gt_6,66,80,191).  visible(mws1gt_6,191).
orientation(mws1gt_5,145,192). coords(mws1gt_5,53,99,192).  visible(mws1gt_5,192). orientation(mws1gt_6,140,192). coords(mws1gt_6,66,81,192).  visible(mws1gt_6,192).
orientation(mws1gt_5,145,193). coords(mws1gt_5,53,99,193).  visible(mws1gt_5,193). orientation(mws1gt_6,140,193). coords(mws1gt_6,66,81,193).  visible(mws1gt_6,193).
happensAt(disappear(mws1gt_5),192). happensAt(disappear(mws1gt_6),192).
1{simpleEvent(T,mws1gt_5,active); simpleEvent(T,mws1gt_5,inactive); simpleEvent(T,mws1gt_5,walking); simpleEvent(T,mws1gt_5,running)}1 :- T=168..192.
1{simpleEvent(T,mws1gt_6,active); simpleEvent(T,mws1gt_6,inactive); simpleEvent(T,mws1gt_6,walking); simpleEvent(T,mws1gt_6,running)}1 :- T=168..192.
:- holdsAt(moving(mws1gt_5,mws1gt_6),168). :- holdsAt(moving(mws1gt_6,mws1gt_5),168).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),169). :- not holdsAt(moving(mws1gt_6,mws1gt_5),169).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),170). :- not holdsAt(moving(mws1gt_6,mws1gt_5),170).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),171). :- not holdsAt(moving(mws1gt_6,mws1gt_5),171).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),172). :- not holdsAt(moving(mws1gt_6,mws1gt_5),172).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),173). :- not holdsAt(moving(mws1gt_6,mws1gt_5),173).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),174). :- not holdsAt(moving(mws1gt_6,mws1gt_5),174).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),175). :- not holdsAt(moving(mws1gt_6,mws1gt_5),175).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),176). :- not holdsAt(moving(mws1gt_6,mws1gt_5),176).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),177). :- not holdsAt(moving(mws1gt_6,mws1gt_5),177).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),178). :- not holdsAt(moving(mws1gt_6,mws1gt_5),178).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),179). :- not holdsAt(moving(mws1gt_6,mws1gt_5),179).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),180). :- not holdsAt(moving(mws1gt_6,mws1gt_5),180).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),181). :- not holdsAt(moving(mws1gt_6,mws1gt_5),181).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),182). :- not holdsAt(moving(mws1gt_6,mws1gt_5),182).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),183). :- not holdsAt(moving(mws1gt_6,mws1gt_5),183).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),184). :- not holdsAt(moving(mws1gt_6,mws1gt_5),184).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),185). :- not holdsAt(moving(mws1gt_6,mws1gt_5),185).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),186). :- not holdsAt(moving(mws1gt_6,mws1gt_5),186).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),187). :- not holdsAt(moving(mws1gt_6,mws1gt_5),187).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),188). :- not holdsAt(moving(mws1gt_6,mws1gt_5),188).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),189). :- not holdsAt(moving(mws1gt_6,mws1gt_5),189).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),190). :- not holdsAt(moving(mws1gt_6,mws1gt_5),190).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),191). :- not holdsAt(moving(mws1gt_6,mws1gt_5),191).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),192). :- not holdsAt(moving(mws1gt_6,mws1gt_5),192).
:- holdsAt(moving(mws1gt_5,mws1gt_6),193). :- holdsAt(moving(mws1gt_6,mws1gt_5),193).
:- simpleEvent(T,P,inactive), T>168 , T<192.
:- simpleEvent(T,P,running), T>168 , T<192.

#show simpleEvent/3.