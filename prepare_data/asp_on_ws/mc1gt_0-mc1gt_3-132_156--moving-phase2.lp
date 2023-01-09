

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
orientation(mc1gt_0,147,132). coords(mc1gt_0,89,104,132).  visible(mc1gt_0,132). orientation(mc1gt_3,146,132). coords(mc1gt_3,86,77,132).  visible(mc1gt_3,132).
orientation(mc1gt_0,147,133). coords(mc1gt_0,90,105,133).  visible(mc1gt_0,133). orientation(mc1gt_3,146,133). coords(mc1gt_3,87,79,133).  visible(mc1gt_3,133).
orientation(mc1gt_0,147,134). coords(mc1gt_0,90,105,134).  visible(mc1gt_0,134). orientation(mc1gt_3,146,134). coords(mc1gt_3,89,79,134).  visible(mc1gt_3,134).
orientation(mc1gt_0,147,135). coords(mc1gt_0,91,106,135).  visible(mc1gt_0,135). orientation(mc1gt_3,146,135). coords(mc1gt_3,89,78,135).  visible(mc1gt_3,135).
orientation(mc1gt_0,147,136). coords(mc1gt_0,92,106,136).  visible(mc1gt_0,136). orientation(mc1gt_3,146,136). coords(mc1gt_3,89,78,136).  visible(mc1gt_3,136).
orientation(mc1gt_0,147,137). coords(mc1gt_0,93,106,137).  visible(mc1gt_0,137). orientation(mc1gt_3,146,137). coords(mc1gt_3,90,78,137).  visible(mc1gt_3,137).
orientation(mc1gt_0,147,138). coords(mc1gt_0,94,106,138).  visible(mc1gt_0,138). orientation(mc1gt_3,146,138). coords(mc1gt_3,91,79,138).  visible(mc1gt_3,138).
orientation(mc1gt_0,147,139). coords(mc1gt_0,95,106,139).  visible(mc1gt_0,139). orientation(mc1gt_3,146,139). coords(mc1gt_3,91,79,139).  visible(mc1gt_3,139).
orientation(mc1gt_0,147,140). coords(mc1gt_0,95,106,140).  visible(mc1gt_0,140). orientation(mc1gt_3,146,140). coords(mc1gt_3,91,79,140).  visible(mc1gt_3,140).
orientation(mc1gt_0,147,141). coords(mc1gt_0,95,106,141).  visible(mc1gt_0,141). orientation(mc1gt_3,146,141). coords(mc1gt_3,91,79,141).  visible(mc1gt_3,141).
orientation(mc1gt_0,147,142). coords(mc1gt_0,95,106,142).  visible(mc1gt_0,142). orientation(mc1gt_3,146,142). coords(mc1gt_3,91,79,142).  visible(mc1gt_3,142).
orientation(mc1gt_0,147,143). coords(mc1gt_0,95,106,143).  visible(mc1gt_0,143). orientation(mc1gt_3,146,143). coords(mc1gt_3,91,79,143).  visible(mc1gt_3,143).
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
orientation(mc1gt_0,147,157). coords(mc1gt_0,104,107,157).  visible(mc1gt_0,157). orientation(mc1gt_3,146,157). coords(mc1gt_3,99,80,157).  visible(mc1gt_3,157).
happensAt(disappear(mc1gt_0),156). happensAt(disappear(mc1gt_3),156).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=132..156.
1{simpleEvent(T,mc1gt_3,active); simpleEvent(T,mc1gt_3,inactive); simpleEvent(T,mc1gt_3,walking); simpleEvent(T,mc1gt_3,running)}1 :- T=132..156.
:- holdsAt(moving(mc1gt_0,mc1gt_3),132). :- holdsAt(moving(mc1gt_3,mc1gt_0),132).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),133). :- not holdsAt(moving(mc1gt_3,mc1gt_0),133).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),134). :- not holdsAt(moving(mc1gt_3,mc1gt_0),134).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),135). :- not holdsAt(moving(mc1gt_3,mc1gt_0),135).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),136). :- not holdsAt(moving(mc1gt_3,mc1gt_0),136).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),137). :- not holdsAt(moving(mc1gt_3,mc1gt_0),137).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),138). :- not holdsAt(moving(mc1gt_3,mc1gt_0),138).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),139). :- not holdsAt(moving(mc1gt_3,mc1gt_0),139).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),140). :- not holdsAt(moving(mc1gt_3,mc1gt_0),140).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),141). :- not holdsAt(moving(mc1gt_3,mc1gt_0),141).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),142). :- not holdsAt(moving(mc1gt_3,mc1gt_0),142).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),143). :- not holdsAt(moving(mc1gt_3,mc1gt_0),143).
:- not holdsAt(moving(mc1gt_0,mc1gt_3),144). :- not holdsAt(moving(mc1gt_3,mc1gt_0),144).
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
:- holdsAt(moving(mc1gt_0,mc1gt_3),157). :- holdsAt(moving(mc1gt_3,mc1gt_0),157).
:- simpleEvent(T,P,inactive), T>132 , T<156.
:- simpleEvent(T,P,running), T>132 , T<156.

#show simpleEvent/3.