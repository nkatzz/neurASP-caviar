
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

	orientation(br1gt_2,169,346). coords(br1gt_2,86,207,346).  visible(br1gt_2,346). happensAt(inactive(br1gt_2),346). orientation(br1gt_3,175,346). coords(br1gt_3,28,163,346).  visible(br1gt_3,346). happensAt(walking(br1gt_3),346).
orientation(br1gt_2,169,347). coords(br1gt_2,86,207,347).  visible(br1gt_2,347). happensAt(inactive(br1gt_2),347). orientation(br1gt_3,175,347). coords(br1gt_3,28,163,347).  visible(br1gt_3,347). happensAt(walking(br1gt_3),347).
orientation(br1gt_2,169,348). coords(br1gt_2,86,207,348).  visible(br1gt_2,348). happensAt(inactive(br1gt_2),348). orientation(br1gt_3,175,348). coords(br1gt_3,29,163,348).  visible(br1gt_3,348). happensAt(walking(br1gt_3),348).
orientation(br1gt_2,169,349). coords(br1gt_2,86,207,349).  visible(br1gt_2,349). happensAt(inactive(br1gt_2),349). orientation(br1gt_3,175,349). coords(br1gt_3,29,163,349).  visible(br1gt_3,349). happensAt(walking(br1gt_3),349).
orientation(br1gt_2,169,350). coords(br1gt_2,86,207,350).  visible(br1gt_2,350). happensAt(inactive(br1gt_2),350). orientation(br1gt_3,162,350). coords(br1gt_3,29,163,350).  visible(br1gt_3,350). happensAt(walking(br1gt_3),350).
orientation(br1gt_2,169,351). coords(br1gt_2,86,207,351).  visible(br1gt_2,351). happensAt(inactive(br1gt_2),351). orientation(br1gt_3,162,351). coords(br1gt_3,29,163,351).  visible(br1gt_3,351). happensAt(walking(br1gt_3),351).
orientation(br1gt_2,169,352). coords(br1gt_2,86,207,352).  visible(br1gt_2,352). happensAt(inactive(br1gt_2),352). orientation(br1gt_3,162,352). coords(br1gt_3,29,163,352).  visible(br1gt_3,352). happensAt(walking(br1gt_3),352).
orientation(br1gt_2,169,353). coords(br1gt_2,86,207,353).  visible(br1gt_2,353). happensAt(inactive(br1gt_2),353). orientation(br1gt_3,162,353). coords(br1gt_3,29,163,353).  visible(br1gt_3,353). happensAt(walking(br1gt_3),353).
orientation(br1gt_2,169,354). coords(br1gt_2,86,207,354).  visible(br1gt_2,354). happensAt(inactive(br1gt_2),354). orientation(br1gt_3,162,354). coords(br1gt_3,29,163,354).  visible(br1gt_3,354). happensAt(walking(br1gt_3),354).
orientation(br1gt_2,169,355). coords(br1gt_2,86,207,355).  visible(br1gt_2,355). happensAt(inactive(br1gt_2),355). orientation(br1gt_3,162,355). coords(br1gt_3,29,163,355).  visible(br1gt_3,355). happensAt(walking(br1gt_3),355).
orientation(br1gt_2,169,356). coords(br1gt_2,86,207,356).  visible(br1gt_2,356). happensAt(inactive(br1gt_2),356). orientation(br1gt_3,162,356). coords(br1gt_3,30,163,356).  visible(br1gt_3,356). happensAt(walking(br1gt_3),356).
orientation(br1gt_2,169,357). coords(br1gt_2,86,207,357).  visible(br1gt_2,357). happensAt(inactive(br1gt_2),357). orientation(br1gt_3,162,357). coords(br1gt_3,30,163,357).  visible(br1gt_3,357). happensAt(walking(br1gt_3),357).
orientation(br1gt_2,169,358). coords(br1gt_2,86,207,358).  visible(br1gt_2,358). happensAt(inactive(br1gt_2),358). orientation(br1gt_3,162,358). coords(br1gt_3,30,163,358).  visible(br1gt_3,358). happensAt(walking(br1gt_3),358).
orientation(br1gt_2,169,359). coords(br1gt_2,86,207,359).  visible(br1gt_2,359). happensAt(inactive(br1gt_2),359). orientation(br1gt_3,162,359). coords(br1gt_3,30,163,359).  visible(br1gt_3,359). happensAt(walking(br1gt_3),359).
orientation(br1gt_2,169,360). coords(br1gt_2,86,207,360).  visible(br1gt_2,360). happensAt(inactive(br1gt_2),360). orientation(br1gt_3,162,360). coords(br1gt_3,31,163,360).  visible(br1gt_3,360). happensAt(walking(br1gt_3),360).
orientation(br1gt_2,169,361). coords(br1gt_2,86,207,361).  visible(br1gt_2,361). happensAt(inactive(br1gt_2),361). orientation(br1gt_3,162,361). coords(br1gt_3,32,163,361).  visible(br1gt_3,361). happensAt(walking(br1gt_3),361).
orientation(br1gt_2,169,362). coords(br1gt_2,86,207,362).  visible(br1gt_2,362). happensAt(inactive(br1gt_2),362). orientation(br1gt_3,162,362). coords(br1gt_3,32,161,362).  visible(br1gt_3,362). happensAt(walking(br1gt_3),362).
orientation(br1gt_2,169,363). coords(br1gt_2,86,207,363).  visible(br1gt_2,363). happensAt(inactive(br1gt_2),363). orientation(br1gt_3,162,363). coords(br1gt_3,32,161,363).  visible(br1gt_3,363). happensAt(walking(br1gt_3),363).
orientation(br1gt_2,169,364). coords(br1gt_2,86,207,364).  visible(br1gt_2,364). happensAt(inactive(br1gt_2),364). orientation(br1gt_3,162,364). coords(br1gt_3,32,161,364).  visible(br1gt_3,364). happensAt(walking(br1gt_3),364).
orientation(br1gt_2,169,365). coords(br1gt_2,86,207,365).  visible(br1gt_2,365). happensAt(inactive(br1gt_2),365). orientation(br1gt_3,162,365). coords(br1gt_3,32,161,365).  visible(br1gt_3,365). happensAt(walking(br1gt_3),365).
orientation(br1gt_2,169,366). coords(br1gt_2,86,207,366).  visible(br1gt_2,366). happensAt(inactive(br1gt_2),366). orientation(br1gt_3,162,366). coords(br1gt_3,32,161,366).  visible(br1gt_3,366). happensAt(walking(br1gt_3),366).
orientation(br1gt_2,169,367). coords(br1gt_2,86,207,367).  visible(br1gt_2,367). happensAt(inactive(br1gt_2),367). orientation(br1gt_3,162,367). coords(br1gt_3,32,161,367).  visible(br1gt_3,367). happensAt(walking(br1gt_3),367).
orientation(br1gt_2,169,368). coords(br1gt_2,86,207,368).  visible(br1gt_2,368). happensAt(inactive(br1gt_2),368). orientation(br1gt_3,162,368). coords(br1gt_3,33,159,368).  visible(br1gt_3,368). happensAt(walking(br1gt_3),368).
orientation(br1gt_2,169,369). coords(br1gt_2,86,207,369).  visible(br1gt_2,369). happensAt(inactive(br1gt_2),369). orientation(br1gt_3,162,369). coords(br1gt_3,33,159,369).  visible(br1gt_3,369). happensAt(walking(br1gt_3),369).
orientation(br1gt_2,169,370). coords(br1gt_2,86,207,370).  visible(br1gt_2,370). happensAt(inactive(br1gt_2),370). orientation(br1gt_3,162,370). coords(br1gt_3,34,159,370).  visible(br1gt_3,370). happensAt(walking(br1gt_3),370).
#show holdsAt/2.