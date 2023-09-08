

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
orientation(mc1gt_0,147,144). coords(mc1gt_0,95,106,144).  visible(mc1gt_0,144). orientation(mc1gt_3,146,144). coords(mc1gt_3,91,79,144).  visible(mc1gt_3,144).
orientation(mc1gt_0,147,145). coords(mc1gt_0,95,107,145).  visible(mc1gt_0,145). orientation(mc1gt_3,146,145). coords(mc1gt_3,92,79,145).  visible(mc1gt_3,145).
orientation(mc1gt_0,147,146). coords(mc1gt_0,96,107,146).  visible(mc1gt_0,146). orientation(mc1gt_3,146,146). coords(mc1gt_3,94,79,146).  visible(mc1gt_3,146).
orientation(mc1gt_0,147,147). coords(mc1gt_0,98,106,147).  visible(mc1gt_0,147). orientation(mc1gt_3,146,147). coords(mc1gt_3,95,79,147).  visible(mc1gt_3,147).
orientation(mc1gt_0,147,148). coords(mc1gt_0,100,107,148).  visible(mc1gt_0,148). orientation(mc1gt_3,146,148). coords(mc1gt_3,95,79,148).  visible(mc1gt_3,148).
orientation(mc1gt_0,147,149). coords(mc1gt_0,101,107,149).  visible(mc1gt_0,149). orientation(mc1gt_3,146,149). coords(mc1gt_3,96,79,149).  visible(mc1gt_3,149).
orientation(mc1gt_0,147,150). coords(mc1gt_0,101,107,150).  visible(mc1gt_0,150). orientation(mc1gt_3,146,150). coords(mc1gt_3,96,79,150).  visible(mc1gt_3,150).
orientation(mc1gt_0,147,151). coords(mc1gt_0,102,107,151).  visible(mc1gt_0,151). orientation(mc1gt_3,146,151). coords(mc1gt_3,97,79,151).  visible(mc1gt_3,151).
orientation(mc1gt_0,147,152). coords(mc1gt_0,103,107,152).  visible(mc1gt_0,152). orientation(mc1gt_3,146,152). coords(mc1gt_3,98,79,152).  visible(mc1gt_3,152).
orientation(mc1gt_0,147,153). coords(mc1gt_0,104,107,153).  visible(mc1gt_0,153). orientation(mc1gt_3,146,153). coords(mc1gt_3,98,79,153).  visible(mc1gt_3,153).
orientation(mc1gt_0,147,154). coords(mc1gt_0,104,107,154).  visible(mc1gt_0,154). orientation(mc1gt_3,146,154). coords(mc1gt_3,99,79,154).  visible(mc1gt_3,154).
orientation(mc1gt_0,147,155). coords(mc1gt_0,104,107,155).  visible(mc1gt_0,155). orientation(mc1gt_3,146,155). coords(mc1gt_3,99,79,155).  visible(mc1gt_3,155).
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
happensAt(disappear(mc1gt_0),168). happensAt(disappear(mc1gt_3),168).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=144..168.
1{simpleEvent(T,mc1gt_3,active); simpleEvent(T,mc1gt_3,inactive); simpleEvent(T,mc1gt_3,walking); simpleEvent(T,mc1gt_3,running)}1 :- T=144..168.
:- holdsAt(moving(mc1gt_0,mc1gt_3),144). :- holdsAt(moving(mc1gt_3,mc1gt_0),144).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),145). :- not holdsAt(moving(mc1gt_3,mc1gt_0),145).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),146). :- not holdsAt(moving(mc1gt_3,mc1gt_0),146).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),147). :- not holdsAt(moving(mc1gt_3,mc1gt_0),147).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),148). :- not holdsAt(moving(mc1gt_3,mc1gt_0),148).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),149). :- not holdsAt(moving(mc1gt_3,mc1gt_0),149).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),150). :- not holdsAt(moving(mc1gt_3,mc1gt_0),150).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),151). :- not holdsAt(moving(mc1gt_3,mc1gt_0),151).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),152). :- not holdsAt(moving(mc1gt_3,mc1gt_0),152).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),153). :- not holdsAt(moving(mc1gt_3,mc1gt_0),153).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),154). :- not holdsAt(moving(mc1gt_3,mc1gt_0),154).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),155). :- not holdsAt(moving(mc1gt_3,mc1gt_0),155).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),156). :- not holdsAt(moving(mc1gt_3,mc1gt_0),156).
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
:- holdsAt(moving(mc1gt_0,mc1gt_3),169). :- holdsAt(moving(mc1gt_3,mc1gt_0),169).
:- simpleEvent(T,P,inactive), T>144 , T<168.
:- simpleEvent(T,P,running), T>144 , T<168.

#show simpleEvent/3.