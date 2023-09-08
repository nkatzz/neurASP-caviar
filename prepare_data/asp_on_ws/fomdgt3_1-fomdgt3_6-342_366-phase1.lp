
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

	orientation(fomdgt3_1,150,342). coords(fomdgt3_1,54,100,342).  visible(fomdgt3_1,342). happensAt(inactive(fomdgt3_1),342). orientation(fomdgt3_6,130,342). coords(fomdgt3_6,217,110,342).  visible(fomdgt3_6,342). happensAt(walking(fomdgt3_6),342).
orientation(fomdgt3_1,150,343). coords(fomdgt3_1,54,100,343).  visible(fomdgt3_1,343). happensAt(inactive(fomdgt3_1),343). orientation(fomdgt3_6,130,343). coords(fomdgt3_6,217,109,343).  visible(fomdgt3_6,343). happensAt(walking(fomdgt3_6),343).
orientation(fomdgt3_1,150,344). coords(fomdgt3_1,54,100,344).  visible(fomdgt3_1,344). happensAt(inactive(fomdgt3_1),344). orientation(fomdgt3_6,130,344). coords(fomdgt3_6,217,109,344).  visible(fomdgt3_6,344). happensAt(walking(fomdgt3_6),344).
orientation(fomdgt3_1,150,345). coords(fomdgt3_1,54,100,345).  visible(fomdgt3_1,345). happensAt(inactive(fomdgt3_1),345). orientation(fomdgt3_6,130,345). coords(fomdgt3_6,217,107,345).  visible(fomdgt3_6,345). happensAt(walking(fomdgt3_6),345).
orientation(fomdgt3_1,150,346). coords(fomdgt3_1,54,100,346).  visible(fomdgt3_1,346). happensAt(inactive(fomdgt3_1),346). orientation(fomdgt3_6,130,346). coords(fomdgt3_6,214,106,346).  visible(fomdgt3_6,346). happensAt(walking(fomdgt3_6),346).
orientation(fomdgt3_1,150,347). coords(fomdgt3_1,54,100,347).  visible(fomdgt3_1,347). happensAt(inactive(fomdgt3_1),347). orientation(fomdgt3_6,130,347). coords(fomdgt3_6,212,106,347).  visible(fomdgt3_6,347). happensAt(walking(fomdgt3_6),347).
orientation(fomdgt3_1,150,348). coords(fomdgt3_1,54,100,348).  visible(fomdgt3_1,348). happensAt(inactive(fomdgt3_1),348). orientation(fomdgt3_6,130,348). coords(fomdgt3_6,207,106,348).  visible(fomdgt3_6,348). happensAt(walking(fomdgt3_6),348).
orientation(fomdgt3_1,150,349). coords(fomdgt3_1,54,100,349).  visible(fomdgt3_1,349). happensAt(inactive(fomdgt3_1),349). orientation(fomdgt3_6,150,349). coords(fomdgt3_6,205,106,349).  visible(fomdgt3_6,349). happensAt(walking(fomdgt3_6),349).
orientation(fomdgt3_1,150,350). coords(fomdgt3_1,54,100,350).  visible(fomdgt3_1,350). happensAt(inactive(fomdgt3_1),350). orientation(fomdgt3_6,150,350). coords(fomdgt3_6,200,103,350).  visible(fomdgt3_6,350). happensAt(walking(fomdgt3_6),350).
orientation(fomdgt3_1,150,351). coords(fomdgt3_1,54,100,351).  visible(fomdgt3_1,351). happensAt(inactive(fomdgt3_1),351). orientation(fomdgt3_6,150,351). coords(fomdgt3_6,199,102,351).  visible(fomdgt3_6,351). happensAt(walking(fomdgt3_6),351).
orientation(fomdgt3_1,150,352). coords(fomdgt3_1,54,100,352).  visible(fomdgt3_1,352). happensAt(inactive(fomdgt3_1),352). orientation(fomdgt3_6,150,352). coords(fomdgt3_6,196,100,352).  visible(fomdgt3_6,352). happensAt(walking(fomdgt3_6),352).
orientation(fomdgt3_1,150,353). coords(fomdgt3_1,54,100,353).  visible(fomdgt3_1,353). happensAt(inactive(fomdgt3_1),353). orientation(fomdgt3_6,150,353). coords(fomdgt3_6,193,100,353).  visible(fomdgt3_6,353). happensAt(walking(fomdgt3_6),353).
orientation(fomdgt3_1,150,354). coords(fomdgt3_1,54,100,354).  visible(fomdgt3_1,354). happensAt(inactive(fomdgt3_1),354). orientation(fomdgt3_6,150,354). coords(fomdgt3_6,193,100,354).  visible(fomdgt3_6,354). happensAt(walking(fomdgt3_6),354).
orientation(fomdgt3_1,150,355). coords(fomdgt3_1,54,100,355).  visible(fomdgt3_1,355). happensAt(inactive(fomdgt3_1),355). orientation(fomdgt3_6,150,355). coords(fomdgt3_6,193,93,355).  visible(fomdgt3_6,355). happensAt(walking(fomdgt3_6),355).
orientation(fomdgt3_1,150,356). coords(fomdgt3_1,54,100,356).  visible(fomdgt3_1,356). happensAt(inactive(fomdgt3_1),356). orientation(fomdgt3_6,150,356). coords(fomdgt3_6,186,90,356).  visible(fomdgt3_6,356). happensAt(walking(fomdgt3_6),356).
orientation(fomdgt3_1,150,357). coords(fomdgt3_1,54,100,357).  visible(fomdgt3_1,357). happensAt(inactive(fomdgt3_1),357). orientation(fomdgt3_6,156,357). coords(fomdgt3_6,184,90,357).  visible(fomdgt3_6,357). happensAt(walking(fomdgt3_6),357).
orientation(fomdgt3_1,150,358). coords(fomdgt3_1,54,100,358).  visible(fomdgt3_1,358). happensAt(inactive(fomdgt3_1),358). orientation(fomdgt3_6,156,358). coords(fomdgt3_6,182,90,358).  visible(fomdgt3_6,358). happensAt(walking(fomdgt3_6),358).
orientation(fomdgt3_1,150,359). coords(fomdgt3_1,46,102,359).  visible(fomdgt3_1,359). happensAt(inactive(fomdgt3_1),359). orientation(fomdgt3_6,156,359). coords(fomdgt3_6,181,90,359).  visible(fomdgt3_6,359). happensAt(walking(fomdgt3_6),359).
orientation(fomdgt3_1,150,360). coords(fomdgt3_1,46,102,360).  visible(fomdgt3_1,360). happensAt(inactive(fomdgt3_1),360). orientation(fomdgt3_6,156,360). coords(fomdgt3_6,178,94,360).  visible(fomdgt3_6,360). happensAt(walking(fomdgt3_6),360).
orientation(fomdgt3_1,150,361). coords(fomdgt3_1,46,102,361).  visible(fomdgt3_1,361). happensAt(inactive(fomdgt3_1),361). orientation(fomdgt3_6,156,361). coords(fomdgt3_6,176,94,361).  visible(fomdgt3_6,361). happensAt(walking(fomdgt3_6),361).
orientation(fomdgt3_1,150,362). coords(fomdgt3_1,46,102,362).  visible(fomdgt3_1,362). happensAt(inactive(fomdgt3_1),362). orientation(fomdgt3_6,156,362). coords(fomdgt3_6,170,95,362).  visible(fomdgt3_6,362). happensAt(walking(fomdgt3_6),362).
orientation(fomdgt3_1,150,363). coords(fomdgt3_1,46,102,363).  visible(fomdgt3_1,363). happensAt(inactive(fomdgt3_1),363). orientation(fomdgt3_6,156,363). coords(fomdgt3_6,170,95,363).  visible(fomdgt3_6,363). happensAt(walking(fomdgt3_6),363).
orientation(fomdgt3_1,150,364). coords(fomdgt3_1,46,102,364).  visible(fomdgt3_1,364). happensAt(inactive(fomdgt3_1),364). orientation(fomdgt3_6,156,364). coords(fomdgt3_6,159,95,364).  visible(fomdgt3_6,364). happensAt(running(fomdgt3_6),364).
orientation(fomdgt3_1,150,365). coords(fomdgt3_1,46,102,365).  visible(fomdgt3_1,365). happensAt(inactive(fomdgt3_1),365). orientation(fomdgt3_6,156,365). coords(fomdgt3_6,157,95,365).  visible(fomdgt3_6,365). happensAt(running(fomdgt3_6),365).
orientation(fomdgt3_1,150,366). coords(fomdgt3_1,46,102,366).  visible(fomdgt3_1,366). happensAt(inactive(fomdgt3_1),366). orientation(fomdgt3_6,156,366). coords(fomdgt3_6,155,94,366).  visible(fomdgt3_6,366). happensAt(running(fomdgt3_6),366).
#show holdsAt/2.