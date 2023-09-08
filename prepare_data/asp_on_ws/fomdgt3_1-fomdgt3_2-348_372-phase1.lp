
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

	orientation(fomdgt3_1,150,348). coords(fomdgt3_1,54,100,348).  visible(fomdgt3_1,348). happensAt(inactive(fomdgt3_1),348). orientation(fomdgt3_2,30,348). coords(fomdgt3_2,273,80,348).  visible(fomdgt3_2,348). happensAt(walking(fomdgt3_2),348).
orientation(fomdgt3_1,150,349). coords(fomdgt3_1,54,100,349).  visible(fomdgt3_1,349). happensAt(inactive(fomdgt3_1),349). orientation(fomdgt3_2,30,349). coords(fomdgt3_2,273,80,349).  visible(fomdgt3_2,349). happensAt(walking(fomdgt3_2),349).
orientation(fomdgt3_1,150,350). coords(fomdgt3_1,54,100,350).  visible(fomdgt3_1,350). happensAt(inactive(fomdgt3_1),350). orientation(fomdgt3_2,30,350). coords(fomdgt3_2,271,80,350).  visible(fomdgt3_2,350). happensAt(walking(fomdgt3_2),350).
orientation(fomdgt3_1,150,351). coords(fomdgt3_1,54,100,351).  visible(fomdgt3_1,351). happensAt(inactive(fomdgt3_1),351). orientation(fomdgt3_2,30,351). coords(fomdgt3_2,271,80,351).  visible(fomdgt3_2,351). happensAt(walking(fomdgt3_2),351).
orientation(fomdgt3_1,150,352). coords(fomdgt3_1,54,100,352).  visible(fomdgt3_1,352). happensAt(inactive(fomdgt3_1),352). orientation(fomdgt3_2,30,352). coords(fomdgt3_2,270,81,352).  visible(fomdgt3_2,352). happensAt(walking(fomdgt3_2),352).
orientation(fomdgt3_1,150,353). coords(fomdgt3_1,54,100,353).  visible(fomdgt3_1,353). happensAt(inactive(fomdgt3_1),353). orientation(fomdgt3_2,42,353). coords(fomdgt3_2,270,81,353).  visible(fomdgt3_2,353). happensAt(walking(fomdgt3_2),353).
orientation(fomdgt3_1,150,354). coords(fomdgt3_1,54,100,354).  visible(fomdgt3_1,354). happensAt(inactive(fomdgt3_1),354). orientation(fomdgt3_2,42,354). coords(fomdgt3_2,268,81,354).  visible(fomdgt3_2,354). happensAt(walking(fomdgt3_2),354).
orientation(fomdgt3_1,150,355). coords(fomdgt3_1,54,100,355).  visible(fomdgt3_1,355). happensAt(inactive(fomdgt3_1),355). orientation(fomdgt3_2,42,355). coords(fomdgt3_2,267,82,355).  visible(fomdgt3_2,355). happensAt(walking(fomdgt3_2),355).
orientation(fomdgt3_1,150,356). coords(fomdgt3_1,54,100,356).  visible(fomdgt3_1,356). happensAt(inactive(fomdgt3_1),356). orientation(fomdgt3_2,42,356). coords(fomdgt3_2,265,84,356).  visible(fomdgt3_2,356). happensAt(walking(fomdgt3_2),356).
orientation(fomdgt3_1,150,357). coords(fomdgt3_1,54,100,357).  visible(fomdgt3_1,357). happensAt(inactive(fomdgt3_1),357). orientation(fomdgt3_2,42,357). coords(fomdgt3_2,264,86,357).  visible(fomdgt3_2,357). happensAt(walking(fomdgt3_2),357).
orientation(fomdgt3_1,150,358). coords(fomdgt3_1,54,100,358).  visible(fomdgt3_1,358). happensAt(inactive(fomdgt3_1),358). orientation(fomdgt3_2,42,358). coords(fomdgt3_2,264,88,358).  visible(fomdgt3_2,358). happensAt(walking(fomdgt3_2),358).
orientation(fomdgt3_1,150,359). coords(fomdgt3_1,46,102,359).  visible(fomdgt3_1,359). happensAt(inactive(fomdgt3_1),359). orientation(fomdgt3_2,42,359). coords(fomdgt3_2,264,91,359).  visible(fomdgt3_2,359). happensAt(walking(fomdgt3_2),359).
orientation(fomdgt3_1,150,360). coords(fomdgt3_1,46,102,360).  visible(fomdgt3_1,360). happensAt(inactive(fomdgt3_1),360). orientation(fomdgt3_2,42,360). coords(fomdgt3_2,262,93,360).  visible(fomdgt3_2,360). happensAt(walking(fomdgt3_2),360).
orientation(fomdgt3_1,150,361). coords(fomdgt3_1,46,102,361).  visible(fomdgt3_1,361). happensAt(inactive(fomdgt3_1),361). orientation(fomdgt3_2,42,361). coords(fomdgt3_2,262,93,361).  visible(fomdgt3_2,361). happensAt(walking(fomdgt3_2),361).
orientation(fomdgt3_1,150,362). coords(fomdgt3_1,46,102,362).  visible(fomdgt3_1,362). happensAt(inactive(fomdgt3_1),362). orientation(fomdgt3_2,42,362). coords(fomdgt3_2,262,95,362).  visible(fomdgt3_2,362). happensAt(walking(fomdgt3_2),362).
orientation(fomdgt3_1,150,363). coords(fomdgt3_1,46,102,363).  visible(fomdgt3_1,363). happensAt(inactive(fomdgt3_1),363). orientation(fomdgt3_2,42,363). coords(fomdgt3_2,262,95,363).  visible(fomdgt3_2,363). happensAt(walking(fomdgt3_2),363).
orientation(fomdgt3_1,150,364). coords(fomdgt3_1,46,102,364).  visible(fomdgt3_1,364). happensAt(inactive(fomdgt3_1),364). orientation(fomdgt3_2,42,364). coords(fomdgt3_2,262,95,364).  visible(fomdgt3_2,364). happensAt(inactive(fomdgt3_2),364).
orientation(fomdgt3_1,150,365). coords(fomdgt3_1,46,102,365).  visible(fomdgt3_1,365). happensAt(inactive(fomdgt3_1),365). orientation(fomdgt3_2,42,365). coords(fomdgt3_2,262,95,365).  visible(fomdgt3_2,365). happensAt(inactive(fomdgt3_2),365).
orientation(fomdgt3_1,150,366). coords(fomdgt3_1,46,102,366).  visible(fomdgt3_1,366). happensAt(inactive(fomdgt3_1),366). orientation(fomdgt3_2,42,366). coords(fomdgt3_2,262,95,366).  visible(fomdgt3_2,366). happensAt(inactive(fomdgt3_2),366).
orientation(fomdgt3_1,150,367). coords(fomdgt3_1,46,102,367).  visible(fomdgt3_1,367). happensAt(inactive(fomdgt3_1),367). orientation(fomdgt3_2,42,367). coords(fomdgt3_2,262,95,367).  visible(fomdgt3_2,367). happensAt(inactive(fomdgt3_2),367).
orientation(fomdgt3_1,150,368). coords(fomdgt3_1,46,102,368).  visible(fomdgt3_1,368). happensAt(inactive(fomdgt3_1),368). orientation(fomdgt3_2,42,368). coords(fomdgt3_2,262,95,368).  visible(fomdgt3_2,368). happensAt(inactive(fomdgt3_2),368).
orientation(fomdgt3_1,150,369). coords(fomdgt3_1,46,102,369).  visible(fomdgt3_1,369). happensAt(inactive(fomdgt3_1),369). orientation(fomdgt3_2,42,369). coords(fomdgt3_2,262,95,369).  visible(fomdgt3_2,369). happensAt(inactive(fomdgt3_2),369).
orientation(fomdgt3_1,150,370). coords(fomdgt3_1,46,102,370).  visible(fomdgt3_1,370). happensAt(inactive(fomdgt3_1),370). orientation(fomdgt3_2,42,370). coords(fomdgt3_2,262,95,370).  visible(fomdgt3_2,370). happensAt(inactive(fomdgt3_2),370).
orientation(fomdgt3_1,150,371). coords(fomdgt3_1,46,102,371).  visible(fomdgt3_1,371). happensAt(inactive(fomdgt3_1),371). orientation(fomdgt3_2,42,371). coords(fomdgt3_2,262,97,371).  visible(fomdgt3_2,371). happensAt(inactive(fomdgt3_2),371).
orientation(fomdgt3_1,150,372). coords(fomdgt3_1,46,102,372).  visible(fomdgt3_1,372). happensAt(inactive(fomdgt3_1),372). orientation(fomdgt3_2,42,372). coords(fomdgt3_2,262,96,372).  visible(fomdgt3_2,372). happensAt(inactive(fomdgt3_2),372).
#show holdsAt/2.