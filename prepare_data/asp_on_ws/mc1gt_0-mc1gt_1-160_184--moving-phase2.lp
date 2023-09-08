

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

	person(mc1gt_0;mc1gt_1).
orientation(mc1gt_0,147,160). coords(mc1gt_0,106,109,160).  visible(mc1gt_0,160). orientation(mc1gt_1,145,160). coords(mc1gt_1,130,84,160).  visible(mc1gt_1,160).
orientation(mc1gt_0,147,161). coords(mc1gt_0,107,109,161).  visible(mc1gt_0,161). orientation(mc1gt_1,145,161). coords(mc1gt_1,131,84,161).  visible(mc1gt_1,161).
orientation(mc1gt_0,147,162). coords(mc1gt_0,108,109,162).  visible(mc1gt_0,162). orientation(mc1gt_1,145,162). coords(mc1gt_1,131,84,162).  visible(mc1gt_1,162).
orientation(mc1gt_0,147,163). coords(mc1gt_0,109,110,163).  visible(mc1gt_0,163). orientation(mc1gt_1,145,163). coords(mc1gt_1,131,84,163).  visible(mc1gt_1,163).
orientation(mc1gt_0,147,164). coords(mc1gt_0,110,110,164).  visible(mc1gt_0,164). orientation(mc1gt_1,145,164). coords(mc1gt_1,131,84,164).  visible(mc1gt_1,164).
orientation(mc1gt_0,147,165). coords(mc1gt_0,111,110,165).  visible(mc1gt_0,165). orientation(mc1gt_1,145,165). coords(mc1gt_1,131,85,165).  visible(mc1gt_1,165).
orientation(mc1gt_0,147,166). coords(mc1gt_0,111,110,166).  visible(mc1gt_0,166). orientation(mc1gt_1,145,166). coords(mc1gt_1,131,86,166).  visible(mc1gt_1,166).
orientation(mc1gt_0,147,167). coords(mc1gt_0,111,110,167).  visible(mc1gt_0,167). orientation(mc1gt_1,145,167). coords(mc1gt_1,132,86,167).  visible(mc1gt_1,167).
orientation(mc1gt_0,147,168). coords(mc1gt_0,111,110,168).  visible(mc1gt_0,168). orientation(mc1gt_1,145,168). coords(mc1gt_1,133,87,168).  visible(mc1gt_1,168).
orientation(mc1gt_0,147,169). coords(mc1gt_0,111,110,169).  visible(mc1gt_0,169). orientation(mc1gt_1,145,169). coords(mc1gt_1,134,87,169).  visible(mc1gt_1,169).
orientation(mc1gt_0,147,170). coords(mc1gt_0,112,110,170).  visible(mc1gt_0,170). orientation(mc1gt_1,145,170). coords(mc1gt_1,136,88,170).  visible(mc1gt_1,170).
orientation(mc1gt_0,147,171). coords(mc1gt_0,112,110,171).  visible(mc1gt_0,171). orientation(mc1gt_1,145,171). coords(mc1gt_1,137,89,171).  visible(mc1gt_1,171).
orientation(mc1gt_0,147,172). coords(mc1gt_0,113,110,172).  visible(mc1gt_0,172). orientation(mc1gt_1,145,172). coords(mc1gt_1,138,90,172).  visible(mc1gt_1,172).
orientation(mc1gt_0,147,173). coords(mc1gt_0,114,110,173).  visible(mc1gt_0,173). orientation(mc1gt_1,145,173). coords(mc1gt_1,138,90,173).  visible(mc1gt_1,173).
orientation(mc1gt_0,147,174). coords(mc1gt_0,116,110,174).  visible(mc1gt_0,174). orientation(mc1gt_1,145,174). coords(mc1gt_1,138,90,174).  visible(mc1gt_1,174).
orientation(mc1gt_0,147,175). coords(mc1gt_0,118,111,175).  visible(mc1gt_0,175). orientation(mc1gt_1,145,175). coords(mc1gt_1,139,90,175).  visible(mc1gt_1,175).
orientation(mc1gt_0,147,176). coords(mc1gt_0,119,111,176).  visible(mc1gt_0,176). orientation(mc1gt_1,145,176). coords(mc1gt_1,139,90,176).  visible(mc1gt_1,176).
orientation(mc1gt_0,147,177). coords(mc1gt_0,120,111,177).  visible(mc1gt_0,177). orientation(mc1gt_1,145,177). coords(mc1gt_1,139,91,177).  visible(mc1gt_1,177).
orientation(mc1gt_0,147,178). coords(mc1gt_0,121,111,178).  visible(mc1gt_0,178). orientation(mc1gt_1,145,178). coords(mc1gt_1,139,91,178).  visible(mc1gt_1,178).
orientation(mc1gt_0,147,179). coords(mc1gt_0,121,111,179).  visible(mc1gt_0,179). orientation(mc1gt_1,145,179). coords(mc1gt_1,140,91,179).  visible(mc1gt_1,179).
orientation(mc1gt_0,147,180). coords(mc1gt_0,122,111,180).  visible(mc1gt_0,180). orientation(mc1gt_1,145,180). coords(mc1gt_1,140,91,180).  visible(mc1gt_1,180).
orientation(mc1gt_0,147,181). coords(mc1gt_0,122,111,181).  visible(mc1gt_0,181). orientation(mc1gt_1,145,181). coords(mc1gt_1,140,92,181).  visible(mc1gt_1,181).
orientation(mc1gt_0,147,182). coords(mc1gt_0,122,111,182).  visible(mc1gt_0,182). orientation(mc1gt_1,145,182). coords(mc1gt_1,142,92,182).  visible(mc1gt_1,182).
orientation(mc1gt_0,147,183). coords(mc1gt_0,122,112,183).  visible(mc1gt_0,183). orientation(mc1gt_1,145,183). coords(mc1gt_1,144,92,183).  visible(mc1gt_1,183).
orientation(mc1gt_0,147,184). coords(mc1gt_0,123,112,184).  visible(mc1gt_0,184). orientation(mc1gt_1,145,184). coords(mc1gt_1,146,92,184).  visible(mc1gt_1,184).
orientation(mc1gt_0,147,185). coords(mc1gt_0,123,112,185).  visible(mc1gt_0,185). orientation(mc1gt_1,145,185). coords(mc1gt_1,146,92,185).  visible(mc1gt_1,185).
happensAt(disappear(mc1gt_0),184). happensAt(disappear(mc1gt_1),184).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=160..184.
1{simpleEvent(T,mc1gt_1,active); simpleEvent(T,mc1gt_1,inactive); simpleEvent(T,mc1gt_1,walking); simpleEvent(T,mc1gt_1,running)}1 :- T=160..184.
:- holdsAt(moving(mc1gt_0,mc1gt_1),160). :- holdsAt(moving(mc1gt_1,mc1gt_0),160).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),161). :- not holdsAt(moving(mc1gt_1,mc1gt_0),161).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),162). :- not holdsAt(moving(mc1gt_1,mc1gt_0),162).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),163). :- not holdsAt(moving(mc1gt_1,mc1gt_0),163).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),164). :- not holdsAt(moving(mc1gt_1,mc1gt_0),164).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),165). :- not holdsAt(moving(mc1gt_1,mc1gt_0),165).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),166). :- not holdsAt(moving(mc1gt_1,mc1gt_0),166).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),167). :- not holdsAt(moving(mc1gt_1,mc1gt_0),167).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),168). :- not holdsAt(moving(mc1gt_1,mc1gt_0),168).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),169). :- not holdsAt(moving(mc1gt_1,mc1gt_0),169).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),170). :- not holdsAt(moving(mc1gt_1,mc1gt_0),170).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),171). :- not holdsAt(moving(mc1gt_1,mc1gt_0),171).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),172). :- not holdsAt(moving(mc1gt_1,mc1gt_0),172).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),173). :- not holdsAt(moving(mc1gt_1,mc1gt_0),173).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),174). :- not holdsAt(moving(mc1gt_1,mc1gt_0),174).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),175). :- not holdsAt(moving(mc1gt_1,mc1gt_0),175).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),176). :- not holdsAt(moving(mc1gt_1,mc1gt_0),176).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),177). :- not holdsAt(moving(mc1gt_1,mc1gt_0),177).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),178). :- not holdsAt(moving(mc1gt_1,mc1gt_0),178).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),179). :- not holdsAt(moving(mc1gt_1,mc1gt_0),179).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),180). :- not holdsAt(moving(mc1gt_1,mc1gt_0),180).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),181). :- not holdsAt(moving(mc1gt_1,mc1gt_0),181).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),182). :- not holdsAt(moving(mc1gt_1,mc1gt_0),182).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),183). :- not holdsAt(moving(mc1gt_1,mc1gt_0),183).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),184). :- not holdsAt(moving(mc1gt_1,mc1gt_0),184).
:- holdsAt(moving(mc1gt_0,mc1gt_1),185). :- holdsAt(moving(mc1gt_1,mc1gt_0),185).
:- simpleEvent(T,P,inactive), T>160 , T<184.
:- simpleEvent(T,P,running), T>160 , T<184.

#show simpleEvent/3.