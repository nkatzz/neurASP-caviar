

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

	person(mc1gt_2;mc1gt_3).
orientation(mc1gt_2,144,144). coords(mc1gt_2,73,95,144).  visible(mc1gt_2,144). orientation(mc1gt_3,146,144). coords(mc1gt_3,91,79,144).  visible(mc1gt_3,144).
orientation(mc1gt_2,144,145). coords(mc1gt_2,74,95,145).  visible(mc1gt_2,145). orientation(mc1gt_3,146,145). coords(mc1gt_3,92,79,145).  visible(mc1gt_3,145).
orientation(mc1gt_2,144,146). coords(mc1gt_2,74,95,146).  visible(mc1gt_2,146). orientation(mc1gt_3,146,146). coords(mc1gt_3,94,79,146).  visible(mc1gt_3,146).
orientation(mc1gt_2,144,147). coords(mc1gt_2,74,96,147).  visible(mc1gt_2,147). orientation(mc1gt_3,146,147). coords(mc1gt_3,95,79,147).  visible(mc1gt_3,147).
orientation(mc1gt_2,144,148). coords(mc1gt_2,75,96,148).  visible(mc1gt_2,148). orientation(mc1gt_3,146,148). coords(mc1gt_3,95,79,148).  visible(mc1gt_3,148).
orientation(mc1gt_2,144,149). coords(mc1gt_2,75,98,149).  visible(mc1gt_2,149). orientation(mc1gt_3,146,149). coords(mc1gt_3,96,79,149).  visible(mc1gt_3,149).
orientation(mc1gt_2,144,150). coords(mc1gt_2,78,101,150).  visible(mc1gt_2,150). orientation(mc1gt_3,146,150). coords(mc1gt_3,96,79,150).  visible(mc1gt_3,150).
orientation(mc1gt_2,144,151). coords(mc1gt_2,78,102,151).  visible(mc1gt_2,151). orientation(mc1gt_3,146,151). coords(mc1gt_3,97,79,151).  visible(mc1gt_3,151).
orientation(mc1gt_2,144,152). coords(mc1gt_2,79,103,152).  visible(mc1gt_2,152). orientation(mc1gt_3,146,152). coords(mc1gt_3,98,79,152).  visible(mc1gt_3,152).
orientation(mc1gt_2,144,153). coords(mc1gt_2,79,104,153).  visible(mc1gt_2,153). orientation(mc1gt_3,146,153). coords(mc1gt_3,98,79,153).  visible(mc1gt_3,153).
orientation(mc1gt_2,144,154). coords(mc1gt_2,80,105,154).  visible(mc1gt_2,154). orientation(mc1gt_3,146,154). coords(mc1gt_3,99,79,154).  visible(mc1gt_3,154).
orientation(mc1gt_2,144,155). coords(mc1gt_2,80,105,155).  visible(mc1gt_2,155). orientation(mc1gt_3,146,155). coords(mc1gt_3,99,79,155).  visible(mc1gt_3,155).
orientation(mc1gt_2,144,156). coords(mc1gt_2,80,105,156).  visible(mc1gt_2,156). orientation(mc1gt_3,146,156). coords(mc1gt_3,99,80,156).  visible(mc1gt_3,156).
orientation(mc1gt_2,144,157). coords(mc1gt_2,80,105,157).  visible(mc1gt_2,157). orientation(mc1gt_3,146,157). coords(mc1gt_3,99,81,157).  visible(mc1gt_3,157).
orientation(mc1gt_2,144,158). coords(mc1gt_2,80,106,158).  visible(mc1gt_2,158). orientation(mc1gt_3,146,158). coords(mc1gt_3,99,81,158).  visible(mc1gt_3,158).
orientation(mc1gt_2,144,159). coords(mc1gt_2,80,106,159).  visible(mc1gt_2,159). orientation(mc1gt_3,146,159). coords(mc1gt_3,100,81,159).  visible(mc1gt_3,159).
orientation(mc1gt_2,144,160). coords(mc1gt_2,80,106,160).  visible(mc1gt_2,160). orientation(mc1gt_3,146,160). coords(mc1gt_3,101,81,160).  visible(mc1gt_3,160).
orientation(mc1gt_2,144,161). coords(mc1gt_2,81,106,161).  visible(mc1gt_2,161). orientation(mc1gt_3,146,161). coords(mc1gt_3,102,81,161).  visible(mc1gt_3,161).
orientation(mc1gt_2,144,162). coords(mc1gt_2,81,106,162).  visible(mc1gt_2,162). orientation(mc1gt_3,146,162). coords(mc1gt_3,103,81,162).  visible(mc1gt_3,162).
orientation(mc1gt_2,144,163). coords(mc1gt_2,82,107,163).  visible(mc1gt_2,163). orientation(mc1gt_3,146,163). coords(mc1gt_3,104,81,163).  visible(mc1gt_3,163).
orientation(mc1gt_2,144,164). coords(mc1gt_2,83,107,164).  visible(mc1gt_2,164). orientation(mc1gt_3,146,164). coords(mc1gt_3,104,81,164).  visible(mc1gt_3,164).
orientation(mc1gt_2,144,165). coords(mc1gt_2,84,107,165).  visible(mc1gt_2,165). orientation(mc1gt_3,146,165). coords(mc1gt_3,105,81,165).  visible(mc1gt_3,165).
orientation(mc1gt_2,144,166). coords(mc1gt_2,85,107,166).  visible(mc1gt_2,166). orientation(mc1gt_3,146,166). coords(mc1gt_3,105,81,166).  visible(mc1gt_3,166).
orientation(mc1gt_2,144,167). coords(mc1gt_2,86,108,167).  visible(mc1gt_2,167). orientation(mc1gt_3,146,167). coords(mc1gt_3,105,81,167).  visible(mc1gt_3,167).
orientation(mc1gt_2,144,168). coords(mc1gt_2,87,108,168).  visible(mc1gt_2,168). orientation(mc1gt_3,146,168). coords(mc1gt_3,105,81,168).  visible(mc1gt_3,168).
orientation(mc1gt_2,144,169). coords(mc1gt_2,87,108,169).  visible(mc1gt_2,169). orientation(mc1gt_3,146,169). coords(mc1gt_3,105,81,169).  visible(mc1gt_3,169).
happensAt(disappear(mc1gt_2),168). happensAt(disappear(mc1gt_3),168).
1{simpleEvent(T,mc1gt_2,active); simpleEvent(T,mc1gt_2,inactive); simpleEvent(T,mc1gt_2,walking); simpleEvent(T,mc1gt_2,running)}1 :- T=144..168.
1{simpleEvent(T,mc1gt_3,active); simpleEvent(T,mc1gt_3,inactive); simpleEvent(T,mc1gt_3,walking); simpleEvent(T,mc1gt_3,running)}1 :- T=144..168.
:- holdsAt(moving(mc1gt_2,mc1gt_3),144). :- holdsAt(moving(mc1gt_3,mc1gt_2),144).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),145). :- not holdsAt(moving(mc1gt_3,mc1gt_2),145).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),146). :- not holdsAt(moving(mc1gt_3,mc1gt_2),146).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),147). :- not holdsAt(moving(mc1gt_3,mc1gt_2),147).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),148). :- not holdsAt(moving(mc1gt_3,mc1gt_2),148).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),149). :- not holdsAt(moving(mc1gt_3,mc1gt_2),149).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),150). :- not holdsAt(moving(mc1gt_3,mc1gt_2),150).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),151). :- not holdsAt(moving(mc1gt_3,mc1gt_2),151).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),152). :- not holdsAt(moving(mc1gt_3,mc1gt_2),152).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),153). :- not holdsAt(moving(mc1gt_3,mc1gt_2),153).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),154). :- not holdsAt(moving(mc1gt_3,mc1gt_2),154).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),155). :- not holdsAt(moving(mc1gt_3,mc1gt_2),155).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),156). :- not holdsAt(moving(mc1gt_3,mc1gt_2),156).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),157). :- not holdsAt(moving(mc1gt_3,mc1gt_2),157).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),158). :- not holdsAt(moving(mc1gt_3,mc1gt_2),158).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),159). :- not holdsAt(moving(mc1gt_3,mc1gt_2),159).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),160). :- not holdsAt(moving(mc1gt_3,mc1gt_2),160).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),161). :- not holdsAt(moving(mc1gt_3,mc1gt_2),161).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),162). :- not holdsAt(moving(mc1gt_3,mc1gt_2),162).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),163). :- not holdsAt(moving(mc1gt_3,mc1gt_2),163).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),164). :- not holdsAt(moving(mc1gt_3,mc1gt_2),164).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),165). :- not holdsAt(moving(mc1gt_3,mc1gt_2),165).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),166). :- not holdsAt(moving(mc1gt_3,mc1gt_2),166).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),167). :- not holdsAt(moving(mc1gt_3,mc1gt_2),167).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),168). :- not holdsAt(moving(mc1gt_3,mc1gt_2),168).
:- holdsAt(moving(mc1gt_2,mc1gt_3),169). :- holdsAt(moving(mc1gt_3,mc1gt_2),169).
:- simpleEvent(T,P,inactive), T>144 , T<168.
:- simpleEvent(T,P,running), T>144 , T<168.

#show simpleEvent/3.