
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


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(fcgt_1,147,340). coords(fcgt_1,96,129,340).  visible(fcgt_1,340). happensAt(active(fcgt_1),340). orientation(fcgt_3,164,340). coords(fcgt_3,98,187,340).  visible(fcgt_3,340). happensAt(inactive(fcgt_3),340).
orientation(fcgt_1,150,341). coords(fcgt_1,96,129,341).  visible(fcgt_1,341). happensAt(active(fcgt_1),341). orientation(fcgt_3,164,341). coords(fcgt_3,98,187,341).  visible(fcgt_3,341). happensAt(inactive(fcgt_3),341).
orientation(fcgt_1,148,342). coords(fcgt_1,96,129,342).  visible(fcgt_1,342). happensAt(active(fcgt_1),342). orientation(fcgt_3,164,342). coords(fcgt_3,98,187,342).  visible(fcgt_3,342). happensAt(inactive(fcgt_3),342).
orientation(fcgt_1,146,343). coords(fcgt_1,96,130,343).  visible(fcgt_1,343). happensAt(active(fcgt_1),343). orientation(fcgt_3,164,343). coords(fcgt_3,98,187,343).  visible(fcgt_3,343). happensAt(inactive(fcgt_3),343).
orientation(fcgt_1,146,344). coords(fcgt_1,96,130,344).  visible(fcgt_1,344). happensAt(active(fcgt_1),344). orientation(fcgt_3,164,344). coords(fcgt_3,98,187,344).  visible(fcgt_3,344). happensAt(inactive(fcgt_3),344).
orientation(fcgt_1,146,345). coords(fcgt_1,97,129,345).  visible(fcgt_1,345). happensAt(active(fcgt_1),345). orientation(fcgt_3,164,345). coords(fcgt_3,98,187,345).  visible(fcgt_3,345). happensAt(inactive(fcgt_3),345).
orientation(fcgt_1,146,346). coords(fcgt_1,97,129,346).  visible(fcgt_1,346). happensAt(active(fcgt_1),346). orientation(fcgt_3,164,346). coords(fcgt_3,98,187,346).  visible(fcgt_3,346). happensAt(inactive(fcgt_3),346).
orientation(fcgt_1,141,347). coords(fcgt_1,97,127,347).  visible(fcgt_1,347). happensAt(active(fcgt_1),347). orientation(fcgt_3,164,347). coords(fcgt_3,98,187,347).  visible(fcgt_3,347). happensAt(inactive(fcgt_3),347).
orientation(fcgt_1,141,348). coords(fcgt_1,97,126,348).  visible(fcgt_1,348). happensAt(active(fcgt_1),348). orientation(fcgt_3,164,348). coords(fcgt_3,98,187,348).  visible(fcgt_3,348). happensAt(inactive(fcgt_3),348).
orientation(fcgt_1,141,349). coords(fcgt_1,97,124,349).  visible(fcgt_1,349). happensAt(active(fcgt_1),349). orientation(fcgt_3,164,349). coords(fcgt_3,98,187,349).  visible(fcgt_3,349). happensAt(inactive(fcgt_3),349).
orientation(fcgt_1,141,350). coords(fcgt_1,97,123,350).  visible(fcgt_1,350). happensAt(active(fcgt_1),350). orientation(fcgt_3,164,350). coords(fcgt_3,98,187,350).  visible(fcgt_3,350). happensAt(inactive(fcgt_3),350).
orientation(fcgt_1,141,351). coords(fcgt_1,96,122,351).  visible(fcgt_1,351). happensAt(active(fcgt_1),351). orientation(fcgt_3,164,351). coords(fcgt_3,99,187,351).  visible(fcgt_3,351). happensAt(inactive(fcgt_3),351).
orientation(fcgt_1,141,352). coords(fcgt_1,96,123,352).  visible(fcgt_1,352). happensAt(active(fcgt_1),352). orientation(fcgt_3,164,352). coords(fcgt_3,99,187,352).  visible(fcgt_3,352). happensAt(inactive(fcgt_3),352).
orientation(fcgt_1,141,353). coords(fcgt_1,96,121,353).  visible(fcgt_1,353). happensAt(active(fcgt_1),353). orientation(fcgt_3,164,353). coords(fcgt_3,99,187,353).  visible(fcgt_3,353). happensAt(inactive(fcgt_3),353).
orientation(fcgt_1,141,354). coords(fcgt_1,96,121,354).  visible(fcgt_1,354). happensAt(active(fcgt_1),354). orientation(fcgt_3,164,354). coords(fcgt_3,99,187,354).  visible(fcgt_3,354). happensAt(inactive(fcgt_3),354).
orientation(fcgt_1,141,355). coords(fcgt_1,96,121,355).  visible(fcgt_1,355). happensAt(active(fcgt_1),355). orientation(fcgt_3,164,355). coords(fcgt_3,99,187,355).  visible(fcgt_3,355). happensAt(inactive(fcgt_3),355).
orientation(fcgt_1,141,356). coords(fcgt_1,96,120,356).  visible(fcgt_1,356). happensAt(active(fcgt_1),356). orientation(fcgt_3,164,356). coords(fcgt_3,99,187,356).  visible(fcgt_3,356). happensAt(inactive(fcgt_3),356).
orientation(fcgt_1,141,357). coords(fcgt_1,96,121,357).  visible(fcgt_1,357). happensAt(active(fcgt_1),357). orientation(fcgt_3,164,357). coords(fcgt_3,99,187,357).  visible(fcgt_3,357). happensAt(inactive(fcgt_3),357).
orientation(fcgt_1,141,358). coords(fcgt_1,96,121,358).  visible(fcgt_1,358). happensAt(active(fcgt_1),358). orientation(fcgt_3,164,358). coords(fcgt_3,99,187,358).  visible(fcgt_3,358). happensAt(inactive(fcgt_3),358).
orientation(fcgt_1,141,359). coords(fcgt_1,96,122,359).  visible(fcgt_1,359). happensAt(active(fcgt_1),359). orientation(fcgt_3,164,359). coords(fcgt_3,99,187,359).  visible(fcgt_3,359). happensAt(inactive(fcgt_3),359).
orientation(fcgt_1,141,360). coords(fcgt_1,95,122,360).  visible(fcgt_1,360). happensAt(active(fcgt_1),360). orientation(fcgt_3,164,360). coords(fcgt_3,99,187,360).  visible(fcgt_3,360). happensAt(inactive(fcgt_3),360).
orientation(fcgt_1,141,361). coords(fcgt_1,95,121,361).  visible(fcgt_1,361). happensAt(active(fcgt_1),361). orientation(fcgt_3,164,361). coords(fcgt_3,99,187,361).  visible(fcgt_3,361). happensAt(inactive(fcgt_3),361).
orientation(fcgt_1,141,362). coords(fcgt_1,95,120,362).  visible(fcgt_1,362). happensAt(active(fcgt_1),362). orientation(fcgt_3,164,362). coords(fcgt_3,99,187,362).  visible(fcgt_3,362). happensAt(inactive(fcgt_3),362).
orientation(fcgt_1,141,363). coords(fcgt_1,95,117,363).  visible(fcgt_1,363). happensAt(active(fcgt_1),363). orientation(fcgt_3,164,363). coords(fcgt_3,99,187,363).  visible(fcgt_3,363). happensAt(inactive(fcgt_3),363).
orientation(fcgt_1,141,364). coords(fcgt_1,95,115,364).  visible(fcgt_1,364). happensAt(active(fcgt_1),364). orientation(fcgt_3,164,364). coords(fcgt_3,99,187,364).  visible(fcgt_3,364). happensAt(inactive(fcgt_3),364).
#show holdsAt/2.