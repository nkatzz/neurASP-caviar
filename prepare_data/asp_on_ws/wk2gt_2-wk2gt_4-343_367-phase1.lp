
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

	orientation(wk2gt_2,141,343). coords(wk2gt_2,98,63,343).  visible(wk2gt_2,343). happensAt(inactive(wk2gt_2),343). orientation(wk2gt_4,154,343). coords(wk2gt_4,209,181,343).  visible(wk2gt_4,343). happensAt(walking(wk2gt_4),343).
orientation(wk2gt_2,141,344). coords(wk2gt_2,98,63,344).  visible(wk2gt_2,344). happensAt(inactive(wk2gt_2),344). orientation(wk2gt_4,154,344). coords(wk2gt_4,206,178,344).  visible(wk2gt_4,344). happensAt(walking(wk2gt_4),344).
orientation(wk2gt_2,141,345). coords(wk2gt_2,98,63,345).  visible(wk2gt_2,345). happensAt(inactive(wk2gt_2),345). orientation(wk2gt_4,154,345). coords(wk2gt_4,204,175,345).  visible(wk2gt_4,345). happensAt(walking(wk2gt_4),345).
orientation(wk2gt_2,141,346). coords(wk2gt_2,98,63,346).  visible(wk2gt_2,346). happensAt(inactive(wk2gt_2),346). orientation(wk2gt_4,154,346). coords(wk2gt_4,202,174,346).  visible(wk2gt_4,346). happensAt(walking(wk2gt_4),346).
orientation(wk2gt_2,141,347). coords(wk2gt_2,98,63,347).  visible(wk2gt_2,347). happensAt(inactive(wk2gt_2),347). orientation(wk2gt_4,154,347). coords(wk2gt_4,202,173,347).  visible(wk2gt_4,347). happensAt(walking(wk2gt_4),347).
orientation(wk2gt_2,141,348). coords(wk2gt_2,98,63,348).  visible(wk2gt_2,348). happensAt(inactive(wk2gt_2),348). orientation(wk2gt_4,154,348). coords(wk2gt_4,201,172,348).  visible(wk2gt_4,348). happensAt(walking(wk2gt_4),348).
orientation(wk2gt_2,141,349). coords(wk2gt_2,98,63,349).  visible(wk2gt_2,349). happensAt(inactive(wk2gt_2),349). orientation(wk2gt_4,154,349). coords(wk2gt_4,201,171,349).  visible(wk2gt_4,349). happensAt(walking(wk2gt_4),349).
orientation(wk2gt_2,141,350). coords(wk2gt_2,98,63,350).  visible(wk2gt_2,350). happensAt(inactive(wk2gt_2),350). orientation(wk2gt_4,154,350). coords(wk2gt_4,200,170,350).  visible(wk2gt_4,350). happensAt(walking(wk2gt_4),350).
orientation(wk2gt_2,141,351). coords(wk2gt_2,98,63,351).  visible(wk2gt_2,351). happensAt(inactive(wk2gt_2),351). orientation(wk2gt_4,154,351). coords(wk2gt_4,199,169,351).  visible(wk2gt_4,351). happensAt(walking(wk2gt_4),351).
orientation(wk2gt_2,141,352). coords(wk2gt_2,98,63,352).  visible(wk2gt_2,352). happensAt(inactive(wk2gt_2),352). orientation(wk2gt_4,148,352). coords(wk2gt_4,197,167,352).  visible(wk2gt_4,352). happensAt(walking(wk2gt_4),352).
orientation(wk2gt_2,141,353). coords(wk2gt_2,98,63,353).  visible(wk2gt_2,353). happensAt(inactive(wk2gt_2),353). orientation(wk2gt_4,148,353). coords(wk2gt_4,195,164,353).  visible(wk2gt_4,353). happensAt(walking(wk2gt_4),353).
orientation(wk2gt_2,141,354). coords(wk2gt_2,98,63,354).  visible(wk2gt_2,354). happensAt(inactive(wk2gt_2),354). orientation(wk2gt_4,148,354). coords(wk2gt_4,193,162,354).  visible(wk2gt_4,354). happensAt(walking(wk2gt_4),354).
orientation(wk2gt_2,141,355). coords(wk2gt_2,98,63,355).  visible(wk2gt_2,355). happensAt(inactive(wk2gt_2),355). orientation(wk2gt_4,148,355). coords(wk2gt_4,190,159,355).  visible(wk2gt_4,355). happensAt(walking(wk2gt_4),355).
orientation(wk2gt_2,141,356). coords(wk2gt_2,98,63,356).  visible(wk2gt_2,356). happensAt(inactive(wk2gt_2),356). orientation(wk2gt_4,148,356). coords(wk2gt_4,187,156,356).  visible(wk2gt_4,356). happensAt(walking(wk2gt_4),356).
orientation(wk2gt_2,141,357). coords(wk2gt_2,98,63,357).  visible(wk2gt_2,357). happensAt(inactive(wk2gt_2),357). orientation(wk2gt_4,148,357). coords(wk2gt_4,186,154,357).  visible(wk2gt_4,357). happensAt(walking(wk2gt_4),357).
orientation(wk2gt_2,141,358). coords(wk2gt_2,98,63,358).  visible(wk2gt_2,358). happensAt(inactive(wk2gt_2),358). orientation(wk2gt_4,148,358). coords(wk2gt_4,185,150,358).  visible(wk2gt_4,358). happensAt(walking(wk2gt_4),358).
orientation(wk2gt_2,141,359). coords(wk2gt_2,98,63,359).  visible(wk2gt_2,359). happensAt(inactive(wk2gt_2),359). orientation(wk2gt_4,148,359). coords(wk2gt_4,185,150,359).  visible(wk2gt_4,359). happensAt(walking(wk2gt_4),359).
orientation(wk2gt_2,141,360). coords(wk2gt_2,98,63,360).  visible(wk2gt_2,360). happensAt(inactive(wk2gt_2),360). orientation(wk2gt_4,151,360). coords(wk2gt_4,184,149,360).  visible(wk2gt_4,360). happensAt(walking(wk2gt_4),360).
orientation(wk2gt_2,141,361). coords(wk2gt_2,98,63,361).  visible(wk2gt_2,361). happensAt(inactive(wk2gt_2),361). orientation(wk2gt_4,151,361). coords(wk2gt_4,183,149,361).  visible(wk2gt_4,361). happensAt(walking(wk2gt_4),361).
orientation(wk2gt_2,141,362). coords(wk2gt_2,98,63,362).  visible(wk2gt_2,362). happensAt(inactive(wk2gt_2),362). orientation(wk2gt_4,151,362). coords(wk2gt_4,183,148,362).  visible(wk2gt_4,362). happensAt(walking(wk2gt_4),362).
orientation(wk2gt_2,141,363). coords(wk2gt_2,98,63,363).  visible(wk2gt_2,363). happensAt(inactive(wk2gt_2),363). orientation(wk2gt_4,151,363). coords(wk2gt_4,182,148,363).  visible(wk2gt_4,363). happensAt(walking(wk2gt_4),363).
orientation(wk2gt_2,141,364). coords(wk2gt_2,98,63,364).  visible(wk2gt_2,364). happensAt(inactive(wk2gt_2),364). orientation(wk2gt_4,151,364). coords(wk2gt_4,181,146,364).  visible(wk2gt_4,364). happensAt(walking(wk2gt_4),364).
orientation(wk2gt_2,141,365). coords(wk2gt_2,98,63,365).  visible(wk2gt_2,365). happensAt(inactive(wk2gt_2),365). orientation(wk2gt_4,151,365). coords(wk2gt_4,179,144,365).  visible(wk2gt_4,365). happensAt(walking(wk2gt_4),365).
orientation(wk2gt_2,141,366). coords(wk2gt_2,98,63,366).  visible(wk2gt_2,366). happensAt(inactive(wk2gt_2),366). orientation(wk2gt_4,151,366). coords(wk2gt_4,177,142,366).  visible(wk2gt_4,366). happensAt(walking(wk2gt_4),366).
orientation(wk2gt_2,141,367). coords(wk2gt_2,98,63,367).  visible(wk2gt_2,367). happensAt(inactive(wk2gt_2),367). orientation(wk2gt_4,151,367). coords(wk2gt_4,174,140,367).  visible(wk2gt_4,367). happensAt(walking(wk2gt_4),367).
#show holdsAt/2.