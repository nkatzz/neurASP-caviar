

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

	person(mc1gt_0;mc1gt_3).
orientation(mc1gt_0,147,156). coords(mc1gt_0,104,107,156).  visible(mc1gt_0,156). orientation(mc1gt_3,146,156). coords(mc1gt_3,99,80,156).  visible(mc1gt_3,156).
orientation(mc1gt_0,147,157). coords(mc1gt_0,104,107,157).  visible(mc1gt_0,157). orientation(mc1gt_3,146,157). coords(mc1gt_3,99,81,157).  visible(mc1gt_3,157).
orientation(mc1gt_0,147,158). coords(mc1gt_0,104,108,158).  visible(mc1gt_0,158). orientation(mc1gt_3,146,158). coords(mc1gt_3,99,81,158).  visible(mc1gt_3,158).
orientation(mc1gt_0,147,159). coords(mc1gt_0,105,108,159).  visible(mc1gt_0,159). orientation(mc1gt_3,146,159). coords(mc1gt_3,100,81,159).  visible(mc1gt_3,159).
orientation(mc1gt_0,147,160). coords(mc1gt_0,106,109,160).  visible(mc1gt_0,160). orientation(mc1gt_3,146,160). coords(mc1gt_3,101,81,160).  visible(mc1gt_3,160).
orientation(mc1gt_0,147,161). coords(mc1gt_0,107,109,161).  visible(mc1gt_0,161). orientation(mc1gt_3,146,161). coords(mc1gt_3,102,81,161).  visible(mc1gt_3,161).
orientation(mc1gt_0,147,162). coords(mc1gt_0,108,109,162).  visible(mc1gt_0,162). orientation(mc1gt_3,146,162). coords(mc1gt_3,103,81,162).  visible(mc1gt_3,162).
orientation(mc1gt_0,147,163). coords(mc1gt_0,109,110,163).  visible(mc1gt_0,163). orientation(mc1gt_3,146,163). coords(mc1gt_3,104,81,163).  visible(mc1gt_3,163).
orientation(mc1gt_0,147,164). coords(mc1gt_0,110,110,164).  visible(mc1gt_0,164). orientation(mc1gt_3,146,164). coords(mc1gt_3,104,81,164).  visible(mc1gt_3,164).
orientation(mc1gt_0,147,165). coords(mc1gt_0,111,110,165).  visible(mc1gt_0,165). orientation(mc1gt_3,146,165). coords(mc1gt_3,105,81,165).  visible(mc1gt_3,165).
orientation(mc1gt_0,147,166). coords(mc1gt_0,111,110,166).  visible(mc1gt_0,166). orientation(mc1gt_3,146,166). coords(mc1gt_3,105,81,166).  visible(mc1gt_3,166).
orientation(mc1gt_0,147,167). coords(mc1gt_0,111,110,167).  visible(mc1gt_0,167). orientation(mc1gt_3,146,167). coords(mc1gt_3,105,81,167).  visible(mc1gt_3,167).
orientation(mc1gt_0,147,168). coords(mc1gt_0,111,110,168).  visible(mc1gt_0,168). orientation(mc1gt_3,146,168). coords(mc1gt_3,105,81,168).  visible(mc1gt_3,168).
orientation(mc1gt_0,147,169). coords(mc1gt_0,111,110,169).  visible(mc1gt_0,169). orientation(mc1gt_3,146,169). coords(mc1gt_3,105,81,169).  visible(mc1gt_3,169).
orientation(mc1gt_0,147,170). coords(mc1gt_0,112,110,170).  visible(mc1gt_0,170). orientation(mc1gt_3,146,170). coords(mc1gt_3,105,81,170).  visible(mc1gt_3,170).
orientation(mc1gt_0,147,171). coords(mc1gt_0,112,110,171).  visible(mc1gt_0,171). orientation(mc1gt_3,146,171). coords(mc1gt_3,106,81,171).  visible(mc1gt_3,171).
orientation(mc1gt_0,147,172). coords(mc1gt_0,113,110,172).  visible(mc1gt_0,172). orientation(mc1gt_3,146,172). coords(mc1gt_3,107,81,172).  visible(mc1gt_3,172).
orientation(mc1gt_0,147,173). coords(mc1gt_0,114,110,173).  visible(mc1gt_0,173). orientation(mc1gt_3,146,173). coords(mc1gt_3,108,81,173).  visible(mc1gt_3,173).
orientation(mc1gt_0,147,174). coords(mc1gt_0,116,110,174).  visible(mc1gt_0,174). orientation(mc1gt_3,146,174). coords(mc1gt_3,109,81,174).  visible(mc1gt_3,174).
orientation(mc1gt_0,147,175). coords(mc1gt_0,118,111,175).  visible(mc1gt_0,175). orientation(mc1gt_3,146,175). coords(mc1gt_3,110,81,175).  visible(mc1gt_3,175).
orientation(mc1gt_0,147,176). coords(mc1gt_0,119,111,176).  visible(mc1gt_0,176). orientation(mc1gt_3,146,176). coords(mc1gt_3,112,81,176).  visible(mc1gt_3,176).
orientation(mc1gt_0,147,177). coords(mc1gt_0,120,111,177).  visible(mc1gt_0,177). orientation(mc1gt_3,146,177). coords(mc1gt_3,112,81,177).  visible(mc1gt_3,177).
orientation(mc1gt_0,147,178). coords(mc1gt_0,121,111,178).  visible(mc1gt_0,178). orientation(mc1gt_3,146,178). coords(mc1gt_3,112,81,178).  visible(mc1gt_3,178).
orientation(mc1gt_0,147,179). coords(mc1gt_0,121,111,179).  visible(mc1gt_0,179). orientation(mc1gt_3,146,179). coords(mc1gt_3,113,81,179).  visible(mc1gt_3,179).
orientation(mc1gt_0,147,180). coords(mc1gt_0,122,111,180).  visible(mc1gt_0,180). orientation(mc1gt_3,146,180). coords(mc1gt_3,113,81,180).  visible(mc1gt_3,180).
orientation(mc1gt_0,147,181). coords(mc1gt_0,122,111,181).  visible(mc1gt_0,181). orientation(mc1gt_3,146,181). coords(mc1gt_3,113,81,181).  visible(mc1gt_3,181).
happensAt(disappear(mc1gt_0),180). happensAt(disappear(mc1gt_3),180).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=156..180.
1{simpleEvent(T,mc1gt_3,active); simpleEvent(T,mc1gt_3,inactive); simpleEvent(T,mc1gt_3,walking); simpleEvent(T,mc1gt_3,running)}1 :- T=156..180.
:- holdsAt(moving(mc1gt_0,mc1gt_3),156). :- holdsAt(moving(mc1gt_3,mc1gt_0),156).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),157). :- not holdsAt(moving(mc1gt_3,mc1gt_0),157).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),158). :- not holdsAt(moving(mc1gt_3,mc1gt_0),158).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),159). :- not holdsAt(moving(mc1gt_3,mc1gt_0),159).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),160). :- not holdsAt(moving(mc1gt_3,mc1gt_0),160).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),161). :- not holdsAt(moving(mc1gt_3,mc1gt_0),161).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),162). :- not holdsAt(moving(mc1gt_3,mc1gt_0),162).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),163). :- not holdsAt(moving(mc1gt_3,mc1gt_0),163).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),164). :- not holdsAt(moving(mc1gt_3,mc1gt_0),164).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),165). :- not holdsAt(moving(mc1gt_3,mc1gt_0),165).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),166). :- not holdsAt(moving(mc1gt_3,mc1gt_0),166).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),167). :- not holdsAt(moving(mc1gt_3,mc1gt_0),167).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),168). :- not holdsAt(moving(mc1gt_3,mc1gt_0),168).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),169). :- not holdsAt(moving(mc1gt_3,mc1gt_0),169).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),170). :- not holdsAt(moving(mc1gt_3,mc1gt_0),170).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),171). :- not holdsAt(moving(mc1gt_3,mc1gt_0),171).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),172). :- not holdsAt(moving(mc1gt_3,mc1gt_0),172).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),173). :- not holdsAt(moving(mc1gt_3,mc1gt_0),173).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),174). :- not holdsAt(moving(mc1gt_3,mc1gt_0),174).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),175). :- not holdsAt(moving(mc1gt_3,mc1gt_0),175).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),176). :- not holdsAt(moving(mc1gt_3,mc1gt_0),176).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),177). :- not holdsAt(moving(mc1gt_3,mc1gt_0),177).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),178). :- not holdsAt(moving(mc1gt_3,mc1gt_0),178).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),179). :- not holdsAt(moving(mc1gt_3,mc1gt_0),179).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),180). :- not holdsAt(moving(mc1gt_3,mc1gt_0),180).
:- holdsAt(moving(mc1gt_0,mc1gt_3),181). :- holdsAt(moving(mc1gt_3,mc1gt_0),181).
:- simpleEvent(T,P,inactive), T>156 , T<180.
:- simpleEvent(T,P,running), T>156 , T<180.

#show simpleEvent/3.