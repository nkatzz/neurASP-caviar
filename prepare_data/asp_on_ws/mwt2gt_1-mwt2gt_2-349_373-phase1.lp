
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

	orientation(mwt2gt_1,143,349). coords(mwt2gt_1,134,77,349).  visible(mwt2gt_1,349). happensAt(active(mwt2gt_1),349). orientation(mwt2gt_2,141,349). coords(mwt2gt_2,142,86,349).  visible(mwt2gt_2,349). happensAt(active(mwt2gt_2),349).
orientation(mwt2gt_1,143,350). coords(mwt2gt_1,134,77,350).  visible(mwt2gt_1,350). happensAt(active(mwt2gt_1),350). orientation(mwt2gt_2,141,350). coords(mwt2gt_2,142,86,350).  visible(mwt2gt_2,350). happensAt(active(mwt2gt_2),350).
orientation(mwt2gt_1,143,351). coords(mwt2gt_1,134,77,351).  visible(mwt2gt_1,351). happensAt(active(mwt2gt_1),351). orientation(mwt2gt_2,141,351). coords(mwt2gt_2,142,86,351).  visible(mwt2gt_2,351). happensAt(active(mwt2gt_2),351).
orientation(mwt2gt_1,143,352). coords(mwt2gt_1,134,77,352).  visible(mwt2gt_1,352). happensAt(active(mwt2gt_1),352). orientation(mwt2gt_2,141,352). coords(mwt2gt_2,142,86,352).  visible(mwt2gt_2,352). happensAt(active(mwt2gt_2),352).
orientation(mwt2gt_1,143,353). coords(mwt2gt_1,134,77,353).  visible(mwt2gt_1,353). happensAt(active(mwt2gt_1),353). orientation(mwt2gt_2,141,353). coords(mwt2gt_2,142,86,353).  visible(mwt2gt_2,353). happensAt(active(mwt2gt_2),353).
orientation(mwt2gt_1,143,354). coords(mwt2gt_1,134,77,354).  visible(mwt2gt_1,354). happensAt(active(mwt2gt_1),354). orientation(mwt2gt_2,141,354). coords(mwt2gt_2,142,86,354).  visible(mwt2gt_2,354). happensAt(active(mwt2gt_2),354).
orientation(mwt2gt_1,143,355). coords(mwt2gt_1,134,77,355).  visible(mwt2gt_1,355). happensAt(active(mwt2gt_1),355). orientation(mwt2gt_2,141,355). coords(mwt2gt_2,142,86,355).  visible(mwt2gt_2,355). happensAt(active(mwt2gt_2),355).
orientation(mwt2gt_1,143,356). coords(mwt2gt_1,134,77,356).  visible(mwt2gt_1,356). happensAt(active(mwt2gt_1),356). orientation(mwt2gt_2,141,356). coords(mwt2gt_2,142,86,356).  visible(mwt2gt_2,356). happensAt(active(mwt2gt_2),356).
orientation(mwt2gt_1,143,357). coords(mwt2gt_1,134,77,357).  visible(mwt2gt_1,357). happensAt(active(mwt2gt_1),357). orientation(mwt2gt_2,141,357). coords(mwt2gt_2,142,86,357).  visible(mwt2gt_2,357). happensAt(active(mwt2gt_2),357).
orientation(mwt2gt_1,143,358). coords(mwt2gt_1,133,77,358).  visible(mwt2gt_1,358). happensAt(active(mwt2gt_1),358). orientation(mwt2gt_2,141,358). coords(mwt2gt_2,141,86,358).  visible(mwt2gt_2,358). happensAt(active(mwt2gt_2),358).
orientation(mwt2gt_1,143,359). coords(mwt2gt_1,132,77,359).  visible(mwt2gt_1,359). happensAt(active(mwt2gt_1),359). orientation(mwt2gt_2,141,359). coords(mwt2gt_2,141,86,359).  visible(mwt2gt_2,359). happensAt(active(mwt2gt_2),359).
orientation(mwt2gt_1,143,360). coords(mwt2gt_1,132,77,360).  visible(mwt2gt_1,360). happensAt(walking(mwt2gt_1),360). orientation(mwt2gt_2,141,360). coords(mwt2gt_2,141,87,360).  visible(mwt2gt_2,360). happensAt(active(mwt2gt_2),360).
orientation(mwt2gt_1,143,361). coords(mwt2gt_1,131,77,361).  visible(mwt2gt_1,361). happensAt(walking(mwt2gt_1),361). orientation(mwt2gt_2,143,361). coords(mwt2gt_2,141,88,361).  visible(mwt2gt_2,361). happensAt(walking(mwt2gt_2),361).
orientation(mwt2gt_1,143,362). coords(mwt2gt_1,130,78,362).  visible(mwt2gt_1,362). happensAt(walking(mwt2gt_1),362). orientation(mwt2gt_2,144,362). coords(mwt2gt_2,140,89,362).  visible(mwt2gt_2,362). happensAt(walking(mwt2gt_2),362).
orientation(mwt2gt_1,143,363). coords(mwt2gt_1,129,77,363).  visible(mwt2gt_1,363). happensAt(walking(mwt2gt_1),363). orientation(mwt2gt_2,144,363). coords(mwt2gt_2,140,90,363).  visible(mwt2gt_2,363). happensAt(walking(mwt2gt_2),363).
orientation(mwt2gt_1,143,364). coords(mwt2gt_1,129,78,364).  visible(mwt2gt_1,364). happensAt(walking(mwt2gt_1),364). orientation(mwt2gt_2,144,364). coords(mwt2gt_2,139,92,364).  visible(mwt2gt_2,364). happensAt(walking(mwt2gt_2),364).
orientation(mwt2gt_1,143,365). coords(mwt2gt_1,128,78,365).  visible(mwt2gt_1,365). happensAt(walking(mwt2gt_1),365). orientation(mwt2gt_2,144,365). coords(mwt2gt_2,139,92,365).  visible(mwt2gt_2,365). happensAt(walking(mwt2gt_2),365).
orientation(mwt2gt_1,143,366). coords(mwt2gt_1,127,78,366).  visible(mwt2gt_1,366). happensAt(walking(mwt2gt_1),366). orientation(mwt2gt_2,144,366). coords(mwt2gt_2,138,93,366).  visible(mwt2gt_2,366). happensAt(walking(mwt2gt_2),366).
orientation(mwt2gt_1,143,367). coords(mwt2gt_1,127,78,367).  visible(mwt2gt_1,367). happensAt(walking(mwt2gt_1),367). orientation(mwt2gt_2,144,367). coords(mwt2gt_2,137,93,367).  visible(mwt2gt_2,367). happensAt(walking(mwt2gt_2),367).
orientation(mwt2gt_1,143,368). coords(mwt2gt_1,127,79,368).  visible(mwt2gt_1,368). happensAt(walking(mwt2gt_1),368). orientation(mwt2gt_2,144,368). coords(mwt2gt_2,137,95,368).  visible(mwt2gt_2,368). happensAt(walking(mwt2gt_2),368).
orientation(mwt2gt_1,143,369). coords(mwt2gt_1,127,80,369).  visible(mwt2gt_1,369). happensAt(walking(mwt2gt_1),369). orientation(mwt2gt_2,144,369). coords(mwt2gt_2,135,95,369).  visible(mwt2gt_2,369). happensAt(walking(mwt2gt_2),369).
orientation(mwt2gt_1,145,370). coords(mwt2gt_1,126,80,370).  visible(mwt2gt_1,370). happensAt(walking(mwt2gt_1),370). orientation(mwt2gt_2,144,370). coords(mwt2gt_2,133,95,370).  visible(mwt2gt_2,370). happensAt(walking(mwt2gt_2),370).
orientation(mwt2gt_1,149,371). coords(mwt2gt_1,126,80,371).  visible(mwt2gt_1,371). happensAt(walking(mwt2gt_1),371). orientation(mwt2gt_2,144,371). coords(mwt2gt_2,131,95,371).  visible(mwt2gt_2,371). happensAt(walking(mwt2gt_2),371).
orientation(mwt2gt_1,149,372). coords(mwt2gt_1,126,82,372).  visible(mwt2gt_1,372). happensAt(walking(mwt2gt_1),372). orientation(mwt2gt_2,151,372). coords(mwt2gt_2,129,96,372).  visible(mwt2gt_2,372). happensAt(walking(mwt2gt_2),372).
orientation(mwt2gt_1,151,373). coords(mwt2gt_1,125,83,373).  visible(mwt2gt_1,373). happensAt(walking(mwt2gt_1),373). orientation(mwt2gt_2,151,373). coords(mwt2gt_2,128,96,373).  visible(mwt2gt_2,373). happensAt(walking(mwt2gt_2),373).
#show holdsAt/2.