

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

	person(mc1gt_0;mc1gt_2).
orientation(mc1gt_0,147,153). coords(mc1gt_0,104,107,153).  visible(mc1gt_0,153). orientation(mc1gt_2,144,153). coords(mc1gt_2,79,104,153).  visible(mc1gt_2,153).
orientation(mc1gt_0,147,154). coords(mc1gt_0,104,107,154).  visible(mc1gt_0,154). orientation(mc1gt_2,144,154). coords(mc1gt_2,80,105,154).  visible(mc1gt_2,154).
orientation(mc1gt_0,147,155). coords(mc1gt_0,104,107,155).  visible(mc1gt_0,155). orientation(mc1gt_2,144,155). coords(mc1gt_2,80,105,155).  visible(mc1gt_2,155).
orientation(mc1gt_0,147,156). coords(mc1gt_0,104,107,156).  visible(mc1gt_0,156). orientation(mc1gt_2,144,156). coords(mc1gt_2,80,105,156).  visible(mc1gt_2,156).
orientation(mc1gt_0,147,157). coords(mc1gt_0,104,107,157).  visible(mc1gt_0,157). orientation(mc1gt_2,144,157). coords(mc1gt_2,80,105,157).  visible(mc1gt_2,157).
orientation(mc1gt_0,147,158). coords(mc1gt_0,104,108,158).  visible(mc1gt_0,158). orientation(mc1gt_2,144,158). coords(mc1gt_2,80,106,158).  visible(mc1gt_2,158).
orientation(mc1gt_0,147,159). coords(mc1gt_0,105,108,159).  visible(mc1gt_0,159). orientation(mc1gt_2,144,159). coords(mc1gt_2,80,106,159).  visible(mc1gt_2,159).
orientation(mc1gt_0,147,160). coords(mc1gt_0,106,109,160).  visible(mc1gt_0,160). orientation(mc1gt_2,144,160). coords(mc1gt_2,80,106,160).  visible(mc1gt_2,160).
orientation(mc1gt_0,147,161). coords(mc1gt_0,107,109,161).  visible(mc1gt_0,161). orientation(mc1gt_2,144,161). coords(mc1gt_2,81,106,161).  visible(mc1gt_2,161).
orientation(mc1gt_0,147,162). coords(mc1gt_0,108,109,162).  visible(mc1gt_0,162). orientation(mc1gt_2,144,162). coords(mc1gt_2,81,106,162).  visible(mc1gt_2,162).
orientation(mc1gt_0,147,163). coords(mc1gt_0,109,110,163).  visible(mc1gt_0,163). orientation(mc1gt_2,144,163). coords(mc1gt_2,82,107,163).  visible(mc1gt_2,163).
orientation(mc1gt_0,147,164). coords(mc1gt_0,110,110,164).  visible(mc1gt_0,164). orientation(mc1gt_2,144,164). coords(mc1gt_2,83,107,164).  visible(mc1gt_2,164).
orientation(mc1gt_0,147,165). coords(mc1gt_0,111,110,165).  visible(mc1gt_0,165). orientation(mc1gt_2,144,165). coords(mc1gt_2,84,107,165).  visible(mc1gt_2,165).
orientation(mc1gt_0,147,166). coords(mc1gt_0,111,110,166).  visible(mc1gt_0,166). orientation(mc1gt_2,144,166). coords(mc1gt_2,85,107,166).  visible(mc1gt_2,166).
orientation(mc1gt_0,147,167). coords(mc1gt_0,111,110,167).  visible(mc1gt_0,167). orientation(mc1gt_2,144,167). coords(mc1gt_2,86,108,167).  visible(mc1gt_2,167).
orientation(mc1gt_0,147,168). coords(mc1gt_0,111,110,168).  visible(mc1gt_0,168). orientation(mc1gt_2,144,168). coords(mc1gt_2,87,108,168).  visible(mc1gt_2,168).
orientation(mc1gt_0,147,169). coords(mc1gt_0,111,110,169).  visible(mc1gt_0,169). orientation(mc1gt_2,144,169). coords(mc1gt_2,87,108,169).  visible(mc1gt_2,169).
orientation(mc1gt_0,147,170). coords(mc1gt_0,112,110,170).  visible(mc1gt_0,170). orientation(mc1gt_2,144,170). coords(mc1gt_2,87,109,170).  visible(mc1gt_2,170).
orientation(mc1gt_0,147,171). coords(mc1gt_0,112,110,171).  visible(mc1gt_0,171). orientation(mc1gt_2,144,171). coords(mc1gt_2,87,109,171).  visible(mc1gt_2,171).
orientation(mc1gt_0,147,172). coords(mc1gt_0,113,110,172).  visible(mc1gt_0,172). orientation(mc1gt_2,144,172). coords(mc1gt_2,87,109,172).  visible(mc1gt_2,172).
orientation(mc1gt_0,147,173). coords(mc1gt_0,114,110,173).  visible(mc1gt_0,173). orientation(mc1gt_2,144,173). coords(mc1gt_2,87,109,173).  visible(mc1gt_2,173).
orientation(mc1gt_0,147,174). coords(mc1gt_0,116,110,174).  visible(mc1gt_0,174). orientation(mc1gt_2,144,174). coords(mc1gt_2,87,109,174).  visible(mc1gt_2,174).
orientation(mc1gt_0,147,175). coords(mc1gt_0,118,111,175).  visible(mc1gt_0,175). orientation(mc1gt_2,144,175). coords(mc1gt_2,88,110,175).  visible(mc1gt_2,175).
orientation(mc1gt_0,147,176). coords(mc1gt_0,119,111,176).  visible(mc1gt_0,176). orientation(mc1gt_2,144,176). coords(mc1gt_2,88,111,176).  visible(mc1gt_2,176).
orientation(mc1gt_0,147,177). coords(mc1gt_0,120,111,177).  visible(mc1gt_0,177). orientation(mc1gt_2,144,177). coords(mc1gt_2,88,113,177).  visible(mc1gt_2,177).
orientation(mc1gt_0,147,178). coords(mc1gt_0,120,111,178).  visible(mc1gt_0,178). orientation(mc1gt_2,144,178). coords(mc1gt_2,88,113,178).  visible(mc1gt_2,178).
happensAt(disappear(mc1gt_0),177). happensAt(disappear(mc1gt_2),177).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=153..177.
1{simpleEvent(T,mc1gt_2,active); simpleEvent(T,mc1gt_2,inactive); simpleEvent(T,mc1gt_2,walking); simpleEvent(T,mc1gt_2,running)}1 :- T=153..177.
:- holdsAt(moving(mc1gt_0,mc1gt_2),153). :- holdsAt(moving(mc1gt_2,mc1gt_0),153).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),154). :- not holdsAt(moving(mc1gt_2,mc1gt_0),154).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),155). :- not holdsAt(moving(mc1gt_2,mc1gt_0),155).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),156). :- not holdsAt(moving(mc1gt_2,mc1gt_0),156).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),157). :- not holdsAt(moving(mc1gt_2,mc1gt_0),157).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),158). :- not holdsAt(moving(mc1gt_2,mc1gt_0),158).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),159). :- not holdsAt(moving(mc1gt_2,mc1gt_0),159).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),160). :- not holdsAt(moving(mc1gt_2,mc1gt_0),160).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),161). :- not holdsAt(moving(mc1gt_2,mc1gt_0),161).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),162). :- not holdsAt(moving(mc1gt_2,mc1gt_0),162).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),163). :- not holdsAt(moving(mc1gt_2,mc1gt_0),163).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),164). :- not holdsAt(moving(mc1gt_2,mc1gt_0),164).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),165). :- not holdsAt(moving(mc1gt_2,mc1gt_0),165).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),166). :- not holdsAt(moving(mc1gt_2,mc1gt_0),166).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),167). :- not holdsAt(moving(mc1gt_2,mc1gt_0),167).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),168). :- not holdsAt(moving(mc1gt_2,mc1gt_0),168).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),169). :- not holdsAt(moving(mc1gt_2,mc1gt_0),169).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),170). :- not holdsAt(moving(mc1gt_2,mc1gt_0),170).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),171). :- not holdsAt(moving(mc1gt_2,mc1gt_0),171).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),172). :- not holdsAt(moving(mc1gt_2,mc1gt_0),172).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),173). :- not holdsAt(moving(mc1gt_2,mc1gt_0),173).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),174). :- not holdsAt(moving(mc1gt_2,mc1gt_0),174).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),175). :- not holdsAt(moving(mc1gt_2,mc1gt_0),175).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),176). :- not holdsAt(moving(mc1gt_2,mc1gt_0),176).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),177). :- not holdsAt(moving(mc1gt_2,mc1gt_0),177).
:- holdsAt(moving(mc1gt_0,mc1gt_2),178). :- holdsAt(moving(mc1gt_2,mc1gt_0),178).
:- simpleEvent(T,P,inactive), T>153 , T<177.
:- simpleEvent(T,P,running), T>153 , T<177.

#show simpleEvent/3.