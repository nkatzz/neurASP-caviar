

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

	person(wk2gt_1;wk2gt_2).
orientation(wk2gt_1,132,144). coords(wk2gt_1,114,53,144).  visible(wk2gt_1,144). orientation(wk2gt_2,141,144). coords(wk2gt_2,98,62,144).  visible(wk2gt_2,144).
orientation(wk2gt_1,132,145). coords(wk2gt_1,114,53,145).  visible(wk2gt_1,145). orientation(wk2gt_2,141,145). coords(wk2gt_2,98,62,145).  visible(wk2gt_2,145).
orientation(wk2gt_1,132,146). coords(wk2gt_1,114,53,146).  visible(wk2gt_1,146). orientation(wk2gt_2,141,146). coords(wk2gt_2,98,62,146).  visible(wk2gt_2,146).
orientation(wk2gt_1,132,147). coords(wk2gt_1,114,53,147).  visible(wk2gt_1,147). orientation(wk2gt_2,141,147). coords(wk2gt_2,98,62,147).  visible(wk2gt_2,147).
orientation(wk2gt_1,132,148). coords(wk2gt_1,114,53,148).  visible(wk2gt_1,148). orientation(wk2gt_2,141,148). coords(wk2gt_2,98,62,148).  visible(wk2gt_2,148).
orientation(wk2gt_1,132,149). coords(wk2gt_1,114,53,149).  visible(wk2gt_1,149). orientation(wk2gt_2,141,149). coords(wk2gt_2,98,62,149).  visible(wk2gt_2,149).
orientation(wk2gt_1,132,150). coords(wk2gt_1,114,53,150).  visible(wk2gt_1,150). orientation(wk2gt_2,141,150). coords(wk2gt_2,98,62,150).  visible(wk2gt_2,150).
orientation(wk2gt_1,132,151). coords(wk2gt_1,114,53,151).  visible(wk2gt_1,151). orientation(wk2gt_2,141,151). coords(wk2gt_2,98,62,151).  visible(wk2gt_2,151).
orientation(wk2gt_1,132,152). coords(wk2gt_1,114,53,152).  visible(wk2gt_1,152). orientation(wk2gt_2,141,152). coords(wk2gt_2,98,62,152).  visible(wk2gt_2,152).
orientation(wk2gt_1,132,153). coords(wk2gt_1,114,53,153).  visible(wk2gt_1,153). orientation(wk2gt_2,141,153). coords(wk2gt_2,98,62,153).  visible(wk2gt_2,153).
orientation(wk2gt_1,129,154). coords(wk2gt_1,114,53,154).  visible(wk2gt_1,154). orientation(wk2gt_2,141,154). coords(wk2gt_2,98,62,154).  visible(wk2gt_2,154).
orientation(wk2gt_1,129,155). coords(wk2gt_1,114,53,155).  visible(wk2gt_1,155). orientation(wk2gt_2,141,155). coords(wk2gt_2,98,62,155).  visible(wk2gt_2,155).
orientation(wk2gt_1,129,156). coords(wk2gt_1,114,53,156).  visible(wk2gt_1,156). orientation(wk2gt_2,141,156). coords(wk2gt_2,98,62,156).  visible(wk2gt_2,156).
orientation(wk2gt_1,129,157). coords(wk2gt_1,114,53,157).  visible(wk2gt_1,157). orientation(wk2gt_2,141,157). coords(wk2gt_2,98,62,157).  visible(wk2gt_2,157).
orientation(wk2gt_1,129,158). coords(wk2gt_1,114,53,158).  visible(wk2gt_1,158). orientation(wk2gt_2,141,158). coords(wk2gt_2,98,62,158).  visible(wk2gt_2,158).
orientation(wk2gt_1,129,159). coords(wk2gt_1,115,53,159).  visible(wk2gt_1,159). orientation(wk2gt_2,141,159). coords(wk2gt_2,98,62,159).  visible(wk2gt_2,159).
orientation(wk2gt_1,129,160). coords(wk2gt_1,115,53,160).  visible(wk2gt_1,160). orientation(wk2gt_2,141,160). coords(wk2gt_2,98,62,160).  visible(wk2gt_2,160).
orientation(wk2gt_1,129,161). coords(wk2gt_1,115,53,161).  visible(wk2gt_1,161). orientation(wk2gt_2,141,161). coords(wk2gt_2,98,62,161).  visible(wk2gt_2,161).
orientation(wk2gt_1,129,162). coords(wk2gt_1,115,53,162).  visible(wk2gt_1,162). orientation(wk2gt_2,141,162). coords(wk2gt_2,98,62,162).  visible(wk2gt_2,162).
orientation(wk2gt_1,129,163). coords(wk2gt_1,115,53,163).  visible(wk2gt_1,163). orientation(wk2gt_2,141,163). coords(wk2gt_2,98,62,163).  visible(wk2gt_2,163).
orientation(wk2gt_1,129,164). coords(wk2gt_1,115,53,164).  visible(wk2gt_1,164). orientation(wk2gt_2,141,164). coords(wk2gt_2,98,62,164).  visible(wk2gt_2,164).
orientation(wk2gt_1,129,165). coords(wk2gt_1,115,53,165).  visible(wk2gt_1,165). orientation(wk2gt_2,141,165). coords(wk2gt_2,98,62,165).  visible(wk2gt_2,165).
orientation(wk2gt_1,129,166). coords(wk2gt_1,115,53,166).  visible(wk2gt_1,166). orientation(wk2gt_2,141,166). coords(wk2gt_2,98,62,166).  visible(wk2gt_2,166).
orientation(wk2gt_1,129,167). coords(wk2gt_1,115,53,167).  visible(wk2gt_1,167). orientation(wk2gt_2,141,167). coords(wk2gt_2,98,62,167).  visible(wk2gt_2,167).
orientation(wk2gt_1,129,168). coords(wk2gt_1,115,53,168).  visible(wk2gt_1,168). orientation(wk2gt_2,141,168). coords(wk2gt_2,98,62,168).  visible(wk2gt_2,168).
orientation(wk2gt_1,129,169). coords(wk2gt_1,115,53,169).  visible(wk2gt_1,169). orientation(wk2gt_2,141,169). coords(wk2gt_2,98,62,169).  visible(wk2gt_2,169).
happensAt(disappear(wk2gt_1),168). happensAt(disappear(wk2gt_2),168).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=144..168.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=144..168.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),144). :- holdsAt(meeting(wk2gt_2,wk2gt_1),144).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),145). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),145).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),146). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),146).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),147). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),147).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),148). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),148).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),149). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),149).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),150). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),150).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),151). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),151).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),152). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),152).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),153). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),153).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),154). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),154).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),155). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),155).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),156). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),156).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),157). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),157).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),158). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),158).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),159). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),159).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),160). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),160).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),161). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),161).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),162). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),162).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),163). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),163).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),164). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),164).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),165). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),165).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),166). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),166).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),167). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),167).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),168). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),168).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),169). :- holdsAt(meeting(wk2gt_2,wk2gt_1),169).
:- simpleEvent(T,P,walking), T>144 , T<168.
:- simpleEvent(T,P,running), T>144 , T<168.

#show simpleEvent/3.