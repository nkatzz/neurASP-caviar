
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

	orientation(fcgt_2,155,342). coords(fcgt_2,94,116,342).  visible(fcgt_2,342). happensAt(active(fcgt_2),342). orientation(fcgt_3,164,342). coords(fcgt_3,98,187,342).  visible(fcgt_3,342). happensAt(inactive(fcgt_3),342).
orientation(fcgt_2,155,343). coords(fcgt_2,94,115,343).  visible(fcgt_2,343). happensAt(active(fcgt_2),343). orientation(fcgt_3,164,343). coords(fcgt_3,98,187,343).  visible(fcgt_3,343). happensAt(inactive(fcgt_3),343).
orientation(fcgt_2,155,344). coords(fcgt_2,94,115,344).  visible(fcgt_2,344). happensAt(active(fcgt_2),344). orientation(fcgt_3,164,344). coords(fcgt_3,98,187,344).  visible(fcgt_3,344). happensAt(inactive(fcgt_3),344).
orientation(fcgt_2,155,345). coords(fcgt_2,94,114,345).  visible(fcgt_2,345). happensAt(active(fcgt_2),345). orientation(fcgt_3,164,345). coords(fcgt_3,98,187,345).  visible(fcgt_3,345). happensAt(inactive(fcgt_3),345).
orientation(fcgt_2,155,346). coords(fcgt_2,93,114,346).  visible(fcgt_2,346). happensAt(active(fcgt_2),346). orientation(fcgt_3,164,346). coords(fcgt_3,98,187,346).  visible(fcgt_3,346). happensAt(inactive(fcgt_3),346).
orientation(fcgt_2,155,347). coords(fcgt_2,94,114,347).  visible(fcgt_2,347). happensAt(active(fcgt_2),347). orientation(fcgt_3,164,347). coords(fcgt_3,98,187,347).  visible(fcgt_3,347). happensAt(inactive(fcgt_3),347).
orientation(fcgt_2,155,348). coords(fcgt_2,94,114,348).  visible(fcgt_2,348). happensAt(active(fcgt_2),348). orientation(fcgt_3,164,348). coords(fcgt_3,98,187,348).  visible(fcgt_3,348). happensAt(inactive(fcgt_3),348).
orientation(fcgt_2,155,349). coords(fcgt_2,93,113,349).  visible(fcgt_2,349). happensAt(active(fcgt_2),349). orientation(fcgt_3,164,349). coords(fcgt_3,98,187,349).  visible(fcgt_3,349). happensAt(inactive(fcgt_3),349).
orientation(fcgt_2,154,350). coords(fcgt_2,93,112,350).  visible(fcgt_2,350). happensAt(active(fcgt_2),350). orientation(fcgt_3,164,350). coords(fcgt_3,98,187,350).  visible(fcgt_3,350). happensAt(inactive(fcgt_3),350).
orientation(fcgt_2,149,351). coords(fcgt_2,92,111,351).  visible(fcgt_2,351). happensAt(active(fcgt_2),351). orientation(fcgt_3,164,351). coords(fcgt_3,99,187,351).  visible(fcgt_3,351). happensAt(inactive(fcgt_3),351).
orientation(fcgt_2,147,352). coords(fcgt_2,92,111,352).  visible(fcgt_2,352). happensAt(active(fcgt_2),352). orientation(fcgt_3,164,352). coords(fcgt_3,99,187,352).  visible(fcgt_3,352). happensAt(inactive(fcgt_3),352).
orientation(fcgt_2,148,353). coords(fcgt_2,92,110,353).  visible(fcgt_2,353). happensAt(active(fcgt_2),353). orientation(fcgt_3,164,353). coords(fcgt_3,99,187,353).  visible(fcgt_3,353). happensAt(inactive(fcgt_3),353).
orientation(fcgt_2,146,354). coords(fcgt_2,92,110,354).  visible(fcgt_2,354). happensAt(active(fcgt_2),354). orientation(fcgt_3,164,354). coords(fcgt_3,99,187,354).  visible(fcgt_3,354). happensAt(inactive(fcgt_3),354).
orientation(fcgt_2,147,355). coords(fcgt_2,92,109,355).  visible(fcgt_2,355). happensAt(active(fcgt_2),355). orientation(fcgt_3,164,355). coords(fcgt_3,99,187,355).  visible(fcgt_3,355). happensAt(inactive(fcgt_3),355).
orientation(fcgt_2,142,356). coords(fcgt_2,92,107,356).  visible(fcgt_2,356). happensAt(active(fcgt_2),356). orientation(fcgt_3,164,356). coords(fcgt_3,99,187,356).  visible(fcgt_3,356). happensAt(inactive(fcgt_3),356).
orientation(fcgt_2,142,357). coords(fcgt_2,92,106,357).  visible(fcgt_2,357). happensAt(active(fcgt_2),357). orientation(fcgt_3,164,357). coords(fcgt_3,99,187,357).  visible(fcgt_3,357). happensAt(inactive(fcgt_3),357).
orientation(fcgt_2,142,358). coords(fcgt_2,92,104,358).  visible(fcgt_2,358). happensAt(active(fcgt_2),358). orientation(fcgt_3,164,358). coords(fcgt_3,99,187,358).  visible(fcgt_3,358). happensAt(inactive(fcgt_3),358).
orientation(fcgt_2,142,359). coords(fcgt_2,91,104,359).  visible(fcgt_2,359). happensAt(active(fcgt_2),359). orientation(fcgt_3,164,359). coords(fcgt_3,99,187,359).  visible(fcgt_3,359). happensAt(inactive(fcgt_3),359).
orientation(fcgt_2,142,360). coords(fcgt_2,91,103,360).  visible(fcgt_2,360). happensAt(active(fcgt_2),360). orientation(fcgt_3,164,360). coords(fcgt_3,99,187,360).  visible(fcgt_3,360). happensAt(inactive(fcgt_3),360).
orientation(fcgt_2,142,361). coords(fcgt_2,91,102,361).  visible(fcgt_2,361). happensAt(active(fcgt_2),361). orientation(fcgt_3,164,361). coords(fcgt_3,99,187,361).  visible(fcgt_3,361). happensAt(inactive(fcgt_3),361).
orientation(fcgt_2,142,362). coords(fcgt_2,91,102,362).  visible(fcgt_2,362). happensAt(active(fcgt_2),362). orientation(fcgt_3,164,362). coords(fcgt_3,99,187,362).  visible(fcgt_3,362). happensAt(inactive(fcgt_3),362).
orientation(fcgt_2,142,363). coords(fcgt_2,91,102,363).  visible(fcgt_2,363). happensAt(active(fcgt_2),363). orientation(fcgt_3,164,363). coords(fcgt_3,99,187,363).  visible(fcgt_3,363). happensAt(inactive(fcgt_3),363).
orientation(fcgt_2,142,364). coords(fcgt_2,91,102,364).  visible(fcgt_2,364). happensAt(active(fcgt_2),364). orientation(fcgt_3,164,364). coords(fcgt_3,99,187,364).  visible(fcgt_3,364). happensAt(inactive(fcgt_3),364).
orientation(fcgt_2,142,365). coords(fcgt_2,90,101,365).  visible(fcgt_2,365). happensAt(active(fcgt_2),365). orientation(fcgt_3,164,365). coords(fcgt_3,99,187,365).  visible(fcgt_3,365). happensAt(inactive(fcgt_3),365).
orientation(fcgt_2,142,366). coords(fcgt_2,90,101,366).  visible(fcgt_2,366). happensAt(active(fcgt_2),366). orientation(fcgt_3,164,366). coords(fcgt_3,99,187,366).  visible(fcgt_3,366). happensAt(inactive(fcgt_3),366).
#show holdsAt/2.