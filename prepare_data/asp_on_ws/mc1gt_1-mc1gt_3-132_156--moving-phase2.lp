

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

	person(mc1gt_1;mc1gt_3).
orientation(mc1gt_1,145,132). coords(mc1gt_1,112,81,132).  visible(mc1gt_1,132). orientation(mc1gt_3,146,132). coords(mc1gt_3,86,77,132).  visible(mc1gt_3,132).
orientation(mc1gt_1,145,133). coords(mc1gt_1,112,81,133).  visible(mc1gt_1,133). orientation(mc1gt_3,146,133). coords(mc1gt_3,87,79,133).  visible(mc1gt_3,133).
orientation(mc1gt_1,145,134). coords(mc1gt_1,112,81,134).  visible(mc1gt_1,134). orientation(mc1gt_3,146,134). coords(mc1gt_3,89,79,134).  visible(mc1gt_3,134).
orientation(mc1gt_1,145,135). coords(mc1gt_1,113,81,135).  visible(mc1gt_1,135). orientation(mc1gt_3,146,135). coords(mc1gt_3,89,78,135).  visible(mc1gt_3,135).
orientation(mc1gt_1,145,136). coords(mc1gt_1,113,81,136).  visible(mc1gt_1,136). orientation(mc1gt_3,146,136). coords(mc1gt_3,89,78,136).  visible(mc1gt_3,136).
orientation(mc1gt_1,145,137). coords(mc1gt_1,113,81,137).  visible(mc1gt_1,137). orientation(mc1gt_3,146,137). coords(mc1gt_3,90,78,137).  visible(mc1gt_3,137).
orientation(mc1gt_1,145,138). coords(mc1gt_1,114,82,138).  visible(mc1gt_1,138). orientation(mc1gt_3,146,138). coords(mc1gt_3,91,79,138).  visible(mc1gt_3,138).
orientation(mc1gt_1,145,139). coords(mc1gt_1,116,82,139).  visible(mc1gt_1,139). orientation(mc1gt_3,146,139). coords(mc1gt_3,91,79,139).  visible(mc1gt_3,139).
orientation(mc1gt_1,145,140). coords(mc1gt_1,116,82,140).  visible(mc1gt_1,140). orientation(mc1gt_3,146,140). coords(mc1gt_3,91,79,140).  visible(mc1gt_3,140).
orientation(mc1gt_1,145,141). coords(mc1gt_1,117,82,141).  visible(mc1gt_1,141). orientation(mc1gt_3,146,141). coords(mc1gt_3,91,79,141).  visible(mc1gt_3,141).
orientation(mc1gt_1,145,142). coords(mc1gt_1,118,83,142).  visible(mc1gt_1,142). orientation(mc1gt_3,146,142). coords(mc1gt_3,91,79,142).  visible(mc1gt_3,142).
orientation(mc1gt_1,145,143). coords(mc1gt_1,118,83,143).  visible(mc1gt_1,143). orientation(mc1gt_3,146,143). coords(mc1gt_3,91,79,143).  visible(mc1gt_3,143).
orientation(mc1gt_1,145,144). coords(mc1gt_1,119,83,144).  visible(mc1gt_1,144). orientation(mc1gt_3,146,144). coords(mc1gt_3,91,79,144).  visible(mc1gt_3,144).
orientation(mc1gt_1,145,145). coords(mc1gt_1,119,83,145).  visible(mc1gt_1,145). orientation(mc1gt_3,146,145). coords(mc1gt_3,92,79,145).  visible(mc1gt_3,145).
orientation(mc1gt_1,145,146). coords(mc1gt_1,119,83,146).  visible(mc1gt_1,146). orientation(mc1gt_3,146,146). coords(mc1gt_3,94,79,146).  visible(mc1gt_3,146).
orientation(mc1gt_1,145,147). coords(mc1gt_1,120,83,147).  visible(mc1gt_1,147). orientation(mc1gt_3,146,147). coords(mc1gt_3,95,79,147).  visible(mc1gt_3,147).
orientation(mc1gt_1,145,148). coords(mc1gt_1,120,83,148).  visible(mc1gt_1,148). orientation(mc1gt_3,146,148). coords(mc1gt_3,95,79,148).  visible(mc1gt_3,148).
orientation(mc1gt_1,145,149). coords(mc1gt_1,120,83,149).  visible(mc1gt_1,149). orientation(mc1gt_3,146,149). coords(mc1gt_3,96,79,149).  visible(mc1gt_3,149).
orientation(mc1gt_1,145,150). coords(mc1gt_1,120,83,150).  visible(mc1gt_1,150). orientation(mc1gt_3,146,150). coords(mc1gt_3,96,79,150).  visible(mc1gt_3,150).
orientation(mc1gt_1,145,151). coords(mc1gt_1,121,83,151).  visible(mc1gt_1,151). orientation(mc1gt_3,146,151). coords(mc1gt_3,97,79,151).  visible(mc1gt_3,151).
orientation(mc1gt_1,145,152). coords(mc1gt_1,123,83,152).  visible(mc1gt_1,152). orientation(mc1gt_3,146,152). coords(mc1gt_3,98,79,152).  visible(mc1gt_3,152).
orientation(mc1gt_1,145,153). coords(mc1gt_1,124,83,153).  visible(mc1gt_1,153). orientation(mc1gt_3,146,153). coords(mc1gt_3,98,79,153).  visible(mc1gt_3,153).
orientation(mc1gt_1,145,154). coords(mc1gt_1,126,83,154).  visible(mc1gt_1,154). orientation(mc1gt_3,146,154). coords(mc1gt_3,99,79,154).  visible(mc1gt_3,154).
orientation(mc1gt_1,145,155). coords(mc1gt_1,127,83,155).  visible(mc1gt_1,155). orientation(mc1gt_3,146,155). coords(mc1gt_3,99,79,155).  visible(mc1gt_3,155).
orientation(mc1gt_1,145,156). coords(mc1gt_1,128,83,156).  visible(mc1gt_1,156). orientation(mc1gt_3,146,156). coords(mc1gt_3,99,80,156).  visible(mc1gt_3,156).
orientation(mc1gt_1,145,157). coords(mc1gt_1,128,83,157).  visible(mc1gt_1,157). orientation(mc1gt_3,146,157). coords(mc1gt_3,99,80,157).  visible(mc1gt_3,157).
happensAt(disappear(mc1gt_1),156). happensAt(disappear(mc1gt_3),156).
1{simpleEvent(T,mc1gt_1,active); simpleEvent(T,mc1gt_1,inactive); simpleEvent(T,mc1gt_1,walking); simpleEvent(T,mc1gt_1,running)}1 :- T=132..156.
1{simpleEvent(T,mc1gt_3,active); simpleEvent(T,mc1gt_3,inactive); simpleEvent(T,mc1gt_3,walking); simpleEvent(T,mc1gt_3,running)}1 :- T=132..156.
:- holdsAt(moving(mc1gt_1,mc1gt_3),132). :- holdsAt(moving(mc1gt_3,mc1gt_1),132).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),133). :- not holdsAt(moving(mc1gt_3,mc1gt_1),133).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),134). :- not holdsAt(moving(mc1gt_3,mc1gt_1),134).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),135). :- not holdsAt(moving(mc1gt_3,mc1gt_1),135).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),136). :- not holdsAt(moving(mc1gt_3,mc1gt_1),136).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),137). :- not holdsAt(moving(mc1gt_3,mc1gt_1),137).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),138). :- not holdsAt(moving(mc1gt_3,mc1gt_1),138).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),139). :- not holdsAt(moving(mc1gt_3,mc1gt_1),139).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),140). :- not holdsAt(moving(mc1gt_3,mc1gt_1),140).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),141). :- not holdsAt(moving(mc1gt_3,mc1gt_1),141).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),142). :- not holdsAt(moving(mc1gt_3,mc1gt_1),142).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),143). :- not holdsAt(moving(mc1gt_3,mc1gt_1),143).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),144). :- not holdsAt(moving(mc1gt_3,mc1gt_1),144).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),145). :- not holdsAt(moving(mc1gt_3,mc1gt_1),145).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),146). :- not holdsAt(moving(mc1gt_3,mc1gt_1),146).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),147). :- not holdsAt(moving(mc1gt_3,mc1gt_1),147).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),148). :- not holdsAt(moving(mc1gt_3,mc1gt_1),148).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),149). :- not holdsAt(moving(mc1gt_3,mc1gt_1),149).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),150). :- not holdsAt(moving(mc1gt_3,mc1gt_1),150).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),151). :- not holdsAt(moving(mc1gt_3,mc1gt_1),151).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),152). :- not holdsAt(moving(mc1gt_3,mc1gt_1),152).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),153). :- not holdsAt(moving(mc1gt_3,mc1gt_1),153).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),154). :- not holdsAt(moving(mc1gt_3,mc1gt_1),154).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),155). :- not holdsAt(moving(mc1gt_3,mc1gt_1),155).
:- not holdsAt(moving(mc1gt_1,mc1gt_3),156). :- not holdsAt(moving(mc1gt_3,mc1gt_1),156).
:- holdsAt(moving(mc1gt_1,mc1gt_3),157). :- holdsAt(moving(mc1gt_3,mc1gt_1),157).
:- simpleEvent(T,P,inactive), T>132 , T<156.
:- simpleEvent(T,P,running), T>132 , T<156.

#show simpleEvent/3.