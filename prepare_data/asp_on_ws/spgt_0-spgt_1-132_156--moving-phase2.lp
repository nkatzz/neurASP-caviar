

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

	person(spgt_0;spgt_1).
orientation(spgt_0,152,132). coords(spgt_0,95,136,132).  visible(spgt_0,132). orientation(spgt_1,154,132). coords(spgt_1,93,121,132).  visible(spgt_1,132).
orientation(spgt_0,152,133). coords(spgt_0,95,136,133).  visible(spgt_0,133). orientation(spgt_1,154,133). coords(spgt_1,94,121,133).  visible(spgt_1,133).
orientation(spgt_0,148,134). coords(spgt_0,95,135,134).  visible(spgt_0,134). orientation(spgt_1,154,134). coords(spgt_1,94,120,134).  visible(spgt_1,134).
orientation(spgt_0,148,135). coords(spgt_0,96,135,135).  visible(spgt_0,135). orientation(spgt_1,154,135). coords(spgt_1,94,120,135).  visible(spgt_1,135).
orientation(spgt_0,148,136). coords(spgt_0,96,135,136).  visible(spgt_0,136). orientation(spgt_1,154,136). coords(spgt_1,94,119,136).  visible(spgt_1,136).
orientation(spgt_0,152,137). coords(spgt_0,96,133,137).  visible(spgt_0,137). orientation(spgt_1,154,137). coords(spgt_1,95,118,137).  visible(spgt_1,137).
orientation(spgt_0,152,138). coords(spgt_0,96,132,138).  visible(spgt_0,138). orientation(spgt_1,154,138). coords(spgt_1,96,117,138).  visible(spgt_1,138).
orientation(spgt_0,153,139). coords(spgt_0,96,131,139).  visible(spgt_0,139). orientation(spgt_1,154,139). coords(spgt_1,96,117,139).  visible(spgt_1,139).
orientation(spgt_0,151,140). coords(spgt_0,97,130,140).  visible(spgt_0,140). orientation(spgt_1,148,140). coords(spgt_1,96,116,140).  visible(spgt_1,140).
orientation(spgt_0,154,141). coords(spgt_0,99,129,141).  visible(spgt_0,141). orientation(spgt_1,148,141). coords(spgt_1,96,116,141).  visible(spgt_1,141).
orientation(spgt_0,155,142). coords(spgt_0,101,129,142).  visible(spgt_0,142). orientation(spgt_1,148,142). coords(spgt_1,97,114,142).  visible(spgt_1,142).
orientation(spgt_0,155,143). coords(spgt_0,101,128,143).  visible(spgt_0,143). orientation(spgt_1,148,143). coords(spgt_1,98,114,143).  visible(spgt_1,143).
orientation(spgt_0,155,144). coords(spgt_0,103,128,144).  visible(spgt_0,144). orientation(spgt_1,148,144). coords(spgt_1,99,113,144).  visible(spgt_1,144).
orientation(spgt_0,159,145). coords(spgt_0,104,127,145).  visible(spgt_0,145). orientation(spgt_1,148,145). coords(spgt_1,99,113,145).  visible(spgt_1,145).
orientation(spgt_0,156,146). coords(spgt_0,104,127,146).  visible(spgt_0,146). orientation(spgt_1,148,146). coords(spgt_1,100,113,146).  visible(spgt_1,146).
orientation(spgt_0,154,147). coords(spgt_0,105,126,147).  visible(spgt_0,147). orientation(spgt_1,148,147). coords(spgt_1,101,113,147).  visible(spgt_1,147).
orientation(spgt_0,154,148). coords(spgt_0,105,126,148).  visible(spgt_0,148). orientation(spgt_1,148,148). coords(spgt_1,101,112,148).  visible(spgt_1,148).
orientation(spgt_0,154,149). coords(spgt_0,105,126,149).  visible(spgt_0,149). orientation(spgt_1,148,149). coords(spgt_1,102,112,149).  visible(spgt_1,149).
orientation(spgt_0,154,150). coords(spgt_0,106,125,150).  visible(spgt_0,150). orientation(spgt_1,148,150). coords(spgt_1,102,112,150).  visible(spgt_1,150).
orientation(spgt_0,154,151). coords(spgt_0,106,125,151).  visible(spgt_0,151). orientation(spgt_1,148,151). coords(spgt_1,102,111,151).  visible(spgt_1,151).
orientation(spgt_0,154,152). coords(spgt_0,106,124,152).  visible(spgt_0,152). orientation(spgt_1,148,152). coords(spgt_1,102,111,152).  visible(spgt_1,152).
orientation(spgt_0,154,153). coords(spgt_0,106,124,153).  visible(spgt_0,153). orientation(spgt_1,148,153). coords(spgt_1,103,110,153).  visible(spgt_1,153).
orientation(spgt_0,154,154). coords(spgt_0,106,123,154).  visible(spgt_0,154). orientation(spgt_1,148,154). coords(spgt_1,103,110,154).  visible(spgt_1,154).
orientation(spgt_0,154,155). coords(spgt_0,107,123,155).  visible(spgt_0,155). orientation(spgt_1,148,155). coords(spgt_1,103,110,155).  visible(spgt_1,155).
orientation(spgt_0,154,156). coords(spgt_0,107,123,156).  visible(spgt_0,156). orientation(spgt_1,148,156). coords(spgt_1,103,109,156).  visible(spgt_1,156).
orientation(spgt_0,154,157). coords(spgt_0,107,123,157).  visible(spgt_0,157). orientation(spgt_1,148,157). coords(spgt_1,103,109,157).  visible(spgt_1,157).
happensAt(disappear(spgt_0),156). happensAt(disappear(spgt_1),156).
1{simpleEvent(T,spgt_0,active); simpleEvent(T,spgt_0,inactive); simpleEvent(T,spgt_0,walking); simpleEvent(T,spgt_0,running)}1 :- T=132..156.
1{simpleEvent(T,spgt_1,active); simpleEvent(T,spgt_1,inactive); simpleEvent(T,spgt_1,walking); simpleEvent(T,spgt_1,running)}1 :- T=132..156.
:- holdsAt(moving(spgt_0,spgt_1),132). :- holdsAt(moving(spgt_1,spgt_0),132).
:- not holdsAt(moving(spgt_0,spgt_1),133). :- not holdsAt(moving(spgt_1,spgt_0),133).
:- not holdsAt(moving(spgt_0,spgt_1),134). :- not holdsAt(moving(spgt_1,spgt_0),134).
:- not holdsAt(moving(spgt_0,spgt_1),135). :- not holdsAt(moving(spgt_1,spgt_0),135).
:- not holdsAt(moving(spgt_0,spgt_1),136). :- not holdsAt(moving(spgt_1,spgt_0),136).
:- not holdsAt(moving(spgt_0,spgt_1),137). :- not holdsAt(moving(spgt_1,spgt_0),137).
:- not holdsAt(moving(spgt_0,spgt_1),138). :- not holdsAt(moving(spgt_1,spgt_0),138).
:- not holdsAt(moving(spgt_0,spgt_1),139). :- not holdsAt(moving(spgt_1,spgt_0),139).
:- not holdsAt(moving(spgt_0,spgt_1),140). :- not holdsAt(moving(spgt_1,spgt_0),140).
:- not holdsAt(moving(spgt_0,spgt_1),141). :- not holdsAt(moving(spgt_1,spgt_0),141).
:- not holdsAt(moving(spgt_0,spgt_1),142). :- not holdsAt(moving(spgt_1,spgt_0),142).
:- not holdsAt(moving(spgt_0,spgt_1),143). :- not holdsAt(moving(spgt_1,spgt_0),143).
:- not holdsAt(moving(spgt_0,spgt_1),144). :- not holdsAt(moving(spgt_1,spgt_0),144).
:- not holdsAt(moving(spgt_0,spgt_1),145). :- not holdsAt(moving(spgt_1,spgt_0),145).
:- not holdsAt(moving(spgt_0,spgt_1),146). :- not holdsAt(moving(spgt_1,spgt_0),146).
:- not holdsAt(moving(spgt_0,spgt_1),147). :- not holdsAt(moving(spgt_1,spgt_0),147).
:- not holdsAt(moving(spgt_0,spgt_1),148). :- not holdsAt(moving(spgt_1,spgt_0),148).
:- not holdsAt(moving(spgt_0,spgt_1),149). :- not holdsAt(moving(spgt_1,spgt_0),149).
:- not holdsAt(moving(spgt_0,spgt_1),150). :- not holdsAt(moving(spgt_1,spgt_0),150).
:- not holdsAt(moving(spgt_0,spgt_1),151). :- not holdsAt(moving(spgt_1,spgt_0),151).
:- not holdsAt(moving(spgt_0,spgt_1),152). :- not holdsAt(moving(spgt_1,spgt_0),152).
:- not holdsAt(moving(spgt_0,spgt_1),153). :- not holdsAt(moving(spgt_1,spgt_0),153).
:- not holdsAt(moving(spgt_0,spgt_1),154). :- not holdsAt(moving(spgt_1,spgt_0),154).
:- not holdsAt(moving(spgt_0,spgt_1),155). :- not holdsAt(moving(spgt_1,spgt_0),155).
:- not holdsAt(moving(spgt_0,spgt_1),156). :- not holdsAt(moving(spgt_1,spgt_0),156).
:- holdsAt(moving(spgt_0,spgt_1),157). :- holdsAt(moving(spgt_1,spgt_0),157).
:- simpleEvent(T,P,inactive), T>132 , T<156.
:- simpleEvent(T,P,running), T>132 , T<156.

#show simpleEvent/3.