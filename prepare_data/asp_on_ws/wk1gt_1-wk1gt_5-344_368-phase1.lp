
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

	orientation(wk1gt_1,149,344). coords(wk1gt_1,147,137,344).  visible(wk1gt_1,344). happensAt(walking(wk1gt_1),344). orientation(wk1gt_5,162,344). coords(wk1gt_5,50,163,344).  visible(wk1gt_5,344). happensAt(inactive(wk1gt_5),344).
orientation(wk1gt_1,149,345). coords(wk1gt_1,145,136,345).  visible(wk1gt_1,345). happensAt(walking(wk1gt_1),345). orientation(wk1gt_5,162,345). coords(wk1gt_5,50,163,345).  visible(wk1gt_5,345). happensAt(inactive(wk1gt_5),345).
orientation(wk1gt_1,149,346). coords(wk1gt_1,143,134,346).  visible(wk1gt_1,346). happensAt(walking(wk1gt_1),346). orientation(wk1gt_5,162,346). coords(wk1gt_5,50,163,346).  visible(wk1gt_5,346). happensAt(inactive(wk1gt_5),346).
orientation(wk1gt_1,149,347). coords(wk1gt_1,141,133,347).  visible(wk1gt_1,347). happensAt(walking(wk1gt_1),347). orientation(wk1gt_5,162,347). coords(wk1gt_5,50,163,347).  visible(wk1gt_5,347). happensAt(inactive(wk1gt_5),347).
orientation(wk1gt_1,149,348). coords(wk1gt_1,140,133,348).  visible(wk1gt_1,348). happensAt(walking(wk1gt_1),348). orientation(wk1gt_5,162,348). coords(wk1gt_5,50,163,348).  visible(wk1gt_5,348). happensAt(inactive(wk1gt_5),348).
orientation(wk1gt_1,149,349). coords(wk1gt_1,139,133,349).  visible(wk1gt_1,349). happensAt(walking(wk1gt_1),349). orientation(wk1gt_5,162,349). coords(wk1gt_5,50,163,349).  visible(wk1gt_5,349). happensAt(inactive(wk1gt_5),349).
orientation(wk1gt_1,149,350). coords(wk1gt_1,139,132,350).  visible(wk1gt_1,350). happensAt(walking(wk1gt_1),350). orientation(wk1gt_5,162,350). coords(wk1gt_5,50,163,350).  visible(wk1gt_5,350). happensAt(inactive(wk1gt_5),350).
orientation(wk1gt_1,149,351). coords(wk1gt_1,139,132,351).  visible(wk1gt_1,351). happensAt(walking(wk1gt_1),351). orientation(wk1gt_5,162,351). coords(wk1gt_5,50,163,351).  visible(wk1gt_5,351). happensAt(inactive(wk1gt_5),351).
orientation(wk1gt_1,149,352). coords(wk1gt_1,138,132,352).  visible(wk1gt_1,352). happensAt(walking(wk1gt_1),352). orientation(wk1gt_5,162,352). coords(wk1gt_5,50,163,352).  visible(wk1gt_5,352). happensAt(inactive(wk1gt_5),352).
orientation(wk1gt_1,149,353). coords(wk1gt_1,138,131,353).  visible(wk1gt_1,353). happensAt(walking(wk1gt_1),353). orientation(wk1gt_5,162,353). coords(wk1gt_5,50,163,353).  visible(wk1gt_5,353). happensAt(inactive(wk1gt_5),353).
orientation(wk1gt_1,149,354). coords(wk1gt_1,138,131,354).  visible(wk1gt_1,354). happensAt(walking(wk1gt_1),354). orientation(wk1gt_5,162,354). coords(wk1gt_5,50,163,354).  visible(wk1gt_5,354). happensAt(inactive(wk1gt_5),354).
orientation(wk1gt_1,149,355). coords(wk1gt_1,136,130,355).  visible(wk1gt_1,355). happensAt(walking(wk1gt_1),355). orientation(wk1gt_5,162,355). coords(wk1gt_5,50,163,355).  visible(wk1gt_5,355). happensAt(inactive(wk1gt_5),355).
orientation(wk1gt_1,149,356). coords(wk1gt_1,136,130,356).  visible(wk1gt_1,356). happensAt(walking(wk1gt_1),356). orientation(wk1gt_5,162,356). coords(wk1gt_5,50,163,356).  visible(wk1gt_5,356). happensAt(inactive(wk1gt_5),356).
orientation(wk1gt_1,145,357). coords(wk1gt_1,133,128,357).  visible(wk1gt_1,357). happensAt(walking(wk1gt_1),357). orientation(wk1gt_5,162,357). coords(wk1gt_5,50,163,357).  visible(wk1gt_5,357). happensAt(inactive(wk1gt_5),357).
orientation(wk1gt_1,145,358). coords(wk1gt_1,131,126,358).  visible(wk1gt_1,358). happensAt(walking(wk1gt_1),358). orientation(wk1gt_5,162,358). coords(wk1gt_5,50,163,358).  visible(wk1gt_5,358). happensAt(inactive(wk1gt_5),358).
orientation(wk1gt_1,145,359). coords(wk1gt_1,131,126,359).  visible(wk1gt_1,359). happensAt(walking(wk1gt_1),359). orientation(wk1gt_5,162,359). coords(wk1gt_5,50,163,359).  visible(wk1gt_5,359). happensAt(inactive(wk1gt_5),359).
orientation(wk1gt_1,151,360). coords(wk1gt_1,130,126,360).  visible(wk1gt_1,360). happensAt(walking(wk1gt_1),360). orientation(wk1gt_5,162,360). coords(wk1gt_5,50,163,360).  visible(wk1gt_5,360). happensAt(inactive(wk1gt_5),360).
orientation(wk1gt_1,151,361). coords(wk1gt_1,129,124,361).  visible(wk1gt_1,361). happensAt(walking(wk1gt_1),361). orientation(wk1gt_5,162,361). coords(wk1gt_5,50,163,361).  visible(wk1gt_5,361). happensAt(inactive(wk1gt_5),361).
orientation(wk1gt_1,151,362). coords(wk1gt_1,129,124,362).  visible(wk1gt_1,362). happensAt(walking(wk1gt_1),362). orientation(wk1gt_5,162,362). coords(wk1gt_5,50,163,362).  visible(wk1gt_5,362). happensAt(inactive(wk1gt_5),362).
orientation(wk1gt_1,148,363). coords(wk1gt_1,129,124,363).  visible(wk1gt_1,363). happensAt(walking(wk1gt_1),363). orientation(wk1gt_5,162,363). coords(wk1gt_5,50,163,363).  visible(wk1gt_5,363). happensAt(inactive(wk1gt_5),363).
orientation(wk1gt_1,148,364). coords(wk1gt_1,129,123,364).  visible(wk1gt_1,364). happensAt(walking(wk1gt_1),364). orientation(wk1gt_5,162,364). coords(wk1gt_5,51,163,364).  visible(wk1gt_5,364). happensAt(inactive(wk1gt_5),364).
orientation(wk1gt_1,148,365). coords(wk1gt_1,128,123,365).  visible(wk1gt_1,365). happensAt(walking(wk1gt_1),365). orientation(wk1gt_5,162,365). coords(wk1gt_5,50,163,365).  visible(wk1gt_5,365). happensAt(inactive(wk1gt_5),365).
orientation(wk1gt_1,148,366). coords(wk1gt_1,128,123,366).  visible(wk1gt_1,366). happensAt(walking(wk1gt_1),366). orientation(wk1gt_5,162,366). coords(wk1gt_5,50,163,366).  visible(wk1gt_5,366). happensAt(inactive(wk1gt_5),366).
orientation(wk1gt_1,153,367). coords(wk1gt_1,127,122,367).  visible(wk1gt_1,367). happensAt(walking(wk1gt_1),367). orientation(wk1gt_5,162,367). coords(wk1gt_5,50,163,367).  visible(wk1gt_5,367). happensAt(inactive(wk1gt_5),367).
orientation(wk1gt_1,153,368). coords(wk1gt_1,126,121,368).  visible(wk1gt_1,368). happensAt(walking(wk1gt_1),368). orientation(wk1gt_5,162,368). coords(wk1gt_5,51,163,368).  visible(wk1gt_5,368). happensAt(inactive(wk1gt_5),368).
#show holdsAt/2.