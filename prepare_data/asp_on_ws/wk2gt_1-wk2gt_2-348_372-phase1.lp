
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

	orientation(wk2gt_1,125,348). coords(wk2gt_1,115,53,348).  visible(wk2gt_1,348). happensAt(inactive(wk2gt_1),348). orientation(wk2gt_2,141,348). coords(wk2gt_2,98,63,348).  visible(wk2gt_2,348). happensAt(inactive(wk2gt_2),348).
orientation(wk2gt_1,125,349). coords(wk2gt_1,115,53,349).  visible(wk2gt_1,349). happensAt(inactive(wk2gt_1),349). orientation(wk2gt_2,141,349). coords(wk2gt_2,98,63,349).  visible(wk2gt_2,349). happensAt(inactive(wk2gt_2),349).
orientation(wk2gt_1,125,350). coords(wk2gt_1,115,53,350).  visible(wk2gt_1,350). happensAt(inactive(wk2gt_1),350). orientation(wk2gt_2,141,350). coords(wk2gt_2,98,63,350).  visible(wk2gt_2,350). happensAt(inactive(wk2gt_2),350).
orientation(wk2gt_1,125,351). coords(wk2gt_1,115,53,351).  visible(wk2gt_1,351). happensAt(inactive(wk2gt_1),351). orientation(wk2gt_2,141,351). coords(wk2gt_2,98,63,351).  visible(wk2gt_2,351). happensAt(inactive(wk2gt_2),351).
orientation(wk2gt_1,125,352). coords(wk2gt_1,115,53,352).  visible(wk2gt_1,352). happensAt(inactive(wk2gt_1),352). orientation(wk2gt_2,141,352). coords(wk2gt_2,98,63,352).  visible(wk2gt_2,352). happensAt(inactive(wk2gt_2),352).
orientation(wk2gt_1,125,353). coords(wk2gt_1,115,53,353).  visible(wk2gt_1,353). happensAt(inactive(wk2gt_1),353). orientation(wk2gt_2,141,353). coords(wk2gt_2,98,63,353).  visible(wk2gt_2,353). happensAt(inactive(wk2gt_2),353).
orientation(wk2gt_1,125,354). coords(wk2gt_1,115,53,354).  visible(wk2gt_1,354). happensAt(inactive(wk2gt_1),354). orientation(wk2gt_2,141,354). coords(wk2gt_2,98,63,354).  visible(wk2gt_2,354). happensAt(inactive(wk2gt_2),354).
orientation(wk2gt_1,125,355). coords(wk2gt_1,115,53,355).  visible(wk2gt_1,355). happensAt(inactive(wk2gt_1),355). orientation(wk2gt_2,141,355). coords(wk2gt_2,98,63,355).  visible(wk2gt_2,355). happensAt(inactive(wk2gt_2),355).
orientation(wk2gt_1,125,356). coords(wk2gt_1,115,53,356).  visible(wk2gt_1,356). happensAt(inactive(wk2gt_1),356). orientation(wk2gt_2,141,356). coords(wk2gt_2,98,63,356).  visible(wk2gt_2,356). happensAt(inactive(wk2gt_2),356).
orientation(wk2gt_1,125,357). coords(wk2gt_1,115,53,357).  visible(wk2gt_1,357). happensAt(inactive(wk2gt_1),357). orientation(wk2gt_2,141,357). coords(wk2gt_2,98,63,357).  visible(wk2gt_2,357). happensAt(inactive(wk2gt_2),357).
orientation(wk2gt_1,125,358). coords(wk2gt_1,115,53,358).  visible(wk2gt_1,358). happensAt(inactive(wk2gt_1),358). orientation(wk2gt_2,141,358). coords(wk2gt_2,98,63,358).  visible(wk2gt_2,358). happensAt(inactive(wk2gt_2),358).
orientation(wk2gt_1,125,359). coords(wk2gt_1,115,53,359).  visible(wk2gt_1,359). happensAt(inactive(wk2gt_1),359). orientation(wk2gt_2,141,359). coords(wk2gt_2,98,63,359).  visible(wk2gt_2,359). happensAt(inactive(wk2gt_2),359).
orientation(wk2gt_1,125,360). coords(wk2gt_1,115,53,360).  visible(wk2gt_1,360). happensAt(inactive(wk2gt_1),360). orientation(wk2gt_2,141,360). coords(wk2gt_2,98,63,360).  visible(wk2gt_2,360). happensAt(inactive(wk2gt_2),360).
orientation(wk2gt_1,125,361). coords(wk2gt_1,115,53,361).  visible(wk2gt_1,361). happensAt(inactive(wk2gt_1),361). orientation(wk2gt_2,141,361). coords(wk2gt_2,98,63,361).  visible(wk2gt_2,361). happensAt(inactive(wk2gt_2),361).
orientation(wk2gt_1,125,362). coords(wk2gt_1,115,53,362).  visible(wk2gt_1,362). happensAt(inactive(wk2gt_1),362). orientation(wk2gt_2,141,362). coords(wk2gt_2,98,63,362).  visible(wk2gt_2,362). happensAt(inactive(wk2gt_2),362).
orientation(wk2gt_1,125,363). coords(wk2gt_1,115,53,363).  visible(wk2gt_1,363). happensAt(inactive(wk2gt_1),363). orientation(wk2gt_2,141,363). coords(wk2gt_2,98,63,363).  visible(wk2gt_2,363). happensAt(inactive(wk2gt_2),363).
orientation(wk2gt_1,125,364). coords(wk2gt_1,115,53,364).  visible(wk2gt_1,364). happensAt(inactive(wk2gt_1),364). orientation(wk2gt_2,141,364). coords(wk2gt_2,98,63,364).  visible(wk2gt_2,364). happensAt(inactive(wk2gt_2),364).
orientation(wk2gt_1,125,365). coords(wk2gt_1,115,53,365).  visible(wk2gt_1,365). happensAt(inactive(wk2gt_1),365). orientation(wk2gt_2,141,365). coords(wk2gt_2,98,63,365).  visible(wk2gt_2,365). happensAt(inactive(wk2gt_2),365).
orientation(wk2gt_1,125,366). coords(wk2gt_1,115,53,366).  visible(wk2gt_1,366). happensAt(inactive(wk2gt_1),366). orientation(wk2gt_2,141,366). coords(wk2gt_2,98,63,366).  visible(wk2gt_2,366). happensAt(inactive(wk2gt_2),366).
orientation(wk2gt_1,125,367). coords(wk2gt_1,115,53,367).  visible(wk2gt_1,367). happensAt(inactive(wk2gt_1),367). orientation(wk2gt_2,141,367). coords(wk2gt_2,98,63,367).  visible(wk2gt_2,367). happensAt(inactive(wk2gt_2),367).
orientation(wk2gt_1,125,368). coords(wk2gt_1,115,53,368).  visible(wk2gt_1,368). happensAt(inactive(wk2gt_1),368). orientation(wk2gt_2,141,368). coords(wk2gt_2,98,63,368).  visible(wk2gt_2,368). happensAt(inactive(wk2gt_2),368).
orientation(wk2gt_1,125,369). coords(wk2gt_1,115,53,369).  visible(wk2gt_1,369). happensAt(inactive(wk2gt_1),369). orientation(wk2gt_2,141,369). coords(wk2gt_2,98,63,369).  visible(wk2gt_2,369). happensAt(inactive(wk2gt_2),369).
orientation(wk2gt_1,125,370). coords(wk2gt_1,115,53,370).  visible(wk2gt_1,370). happensAt(inactive(wk2gt_1),370). orientation(wk2gt_2,141,370). coords(wk2gt_2,98,63,370).  visible(wk2gt_2,370). happensAt(inactive(wk2gt_2),370).
orientation(wk2gt_1,125,371). coords(wk2gt_1,115,53,371).  visible(wk2gt_1,371). happensAt(inactive(wk2gt_1),371). orientation(wk2gt_2,141,371). coords(wk2gt_2,98,63,371).  visible(wk2gt_2,371). happensAt(inactive(wk2gt_2),371).
orientation(wk2gt_1,125,372). coords(wk2gt_1,115,53,372).  visible(wk2gt_1,372). happensAt(inactive(wk2gt_1),372). orientation(wk2gt_2,141,372). coords(wk2gt_2,98,63,372).  visible(wk2gt_2,372). happensAt(inactive(wk2gt_2),372).
#show holdsAt/2.