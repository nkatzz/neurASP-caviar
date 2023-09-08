
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

	orientation(lb1gt_1,151,348). coords(lb1gt_1,165,169,348).  visible(lb1gt_1,348). happensAt(active(lb1gt_1),348). orientation(lb1gt_2,146,348). coords(lb1gt_2,189,148,348).  visible(lb1gt_2,348). happensAt(active(lb1gt_2),348).
orientation(lb1gt_1,151,349). coords(lb1gt_1,165,169,349).  visible(lb1gt_1,349). happensAt(active(lb1gt_1),349). orientation(lb1gt_2,146,349). coords(lb1gt_2,189,148,349).  visible(lb1gt_2,349). happensAt(active(lb1gt_2),349).
orientation(lb1gt_1,151,350). coords(lb1gt_1,165,169,350).  visible(lb1gt_1,350). happensAt(active(lb1gt_1),350). orientation(lb1gt_2,146,350). coords(lb1gt_2,189,148,350).  visible(lb1gt_2,350). happensAt(active(lb1gt_2),350).
orientation(lb1gt_1,151,351). coords(lb1gt_1,165,169,351).  visible(lb1gt_1,351). happensAt(active(lb1gt_1),351). orientation(lb1gt_2,146,351). coords(lb1gt_2,189,148,351).  visible(lb1gt_2,351). happensAt(active(lb1gt_2),351).
orientation(lb1gt_1,151,352). coords(lb1gt_1,165,170,352).  visible(lb1gt_1,352). happensAt(active(lb1gt_1),352). orientation(lb1gt_2,146,352). coords(lb1gt_2,189,148,352).  visible(lb1gt_2,352). happensAt(active(lb1gt_2),352).
orientation(lb1gt_1,151,353). coords(lb1gt_1,165,170,353).  visible(lb1gt_1,353). happensAt(active(lb1gt_1),353). orientation(lb1gt_2,146,353). coords(lb1gt_2,189,148,353).  visible(lb1gt_2,353). happensAt(active(lb1gt_2),353).
orientation(lb1gt_1,151,354). coords(lb1gt_1,165,170,354).  visible(lb1gt_1,354). happensAt(active(lb1gt_1),354). orientation(lb1gt_2,146,354). coords(lb1gt_2,189,148,354).  visible(lb1gt_2,354). happensAt(active(lb1gt_2),354).
orientation(lb1gt_1,151,355). coords(lb1gt_1,165,170,355).  visible(lb1gt_1,355). happensAt(active(lb1gt_1),355). orientation(lb1gt_2,146,355). coords(lb1gt_2,189,148,355).  visible(lb1gt_2,355). happensAt(active(lb1gt_2),355).
orientation(lb1gt_1,151,356). coords(lb1gt_1,165,170,356).  visible(lb1gt_1,356). happensAt(active(lb1gt_1),356). orientation(lb1gt_2,146,356). coords(lb1gt_2,189,147,356).  visible(lb1gt_2,356). happensAt(active(lb1gt_2),356).
orientation(lb1gt_1,151,357). coords(lb1gt_1,165,170,357).  visible(lb1gt_1,357). happensAt(active(lb1gt_1),357). orientation(lb1gt_2,146,357). coords(lb1gt_2,189,147,357).  visible(lb1gt_2,357). happensAt(active(lb1gt_2),357).
orientation(lb1gt_1,151,358). coords(lb1gt_1,165,170,358).  visible(lb1gt_1,358). happensAt(active(lb1gt_1),358). orientation(lb1gt_2,146,358). coords(lb1gt_2,189,145,358).  visible(lb1gt_2,358). happensAt(active(lb1gt_2),358).
orientation(lb1gt_1,151,359). coords(lb1gt_1,165,170,359).  visible(lb1gt_1,359). happensAt(active(lb1gt_1),359). orientation(lb1gt_2,146,359). coords(lb1gt_2,189,145,359).  visible(lb1gt_2,359). happensAt(active(lb1gt_2),359).
orientation(lb1gt_1,151,360). coords(lb1gt_1,165,170,360).  visible(lb1gt_1,360). happensAt(active(lb1gt_1),360). orientation(lb1gt_2,146,360). coords(lb1gt_2,189,145,360).  visible(lb1gt_2,360). happensAt(active(lb1gt_2),360).
orientation(lb1gt_1,151,361). coords(lb1gt_1,165,170,361).  visible(lb1gt_1,361). happensAt(active(lb1gt_1),361). orientation(lb1gt_2,146,361). coords(lb1gt_2,189,145,361).  visible(lb1gt_2,361). happensAt(active(lb1gt_2),361).
orientation(lb1gt_1,151,362). coords(lb1gt_1,165,170,362).  visible(lb1gt_1,362). happensAt(active(lb1gt_1),362). orientation(lb1gt_2,146,362). coords(lb1gt_2,189,144,362).  visible(lb1gt_2,362). happensAt(active(lb1gt_2),362).
orientation(lb1gt_1,151,363). coords(lb1gt_1,165,170,363).  visible(lb1gt_1,363). happensAt(active(lb1gt_1),363). orientation(lb1gt_2,146,363). coords(lb1gt_2,189,144,363).  visible(lb1gt_2,363). happensAt(active(lb1gt_2),363).
orientation(lb1gt_1,151,364). coords(lb1gt_1,165,170,364).  visible(lb1gt_1,364). happensAt(active(lb1gt_1),364). orientation(lb1gt_2,146,364). coords(lb1gt_2,189,144,364).  visible(lb1gt_2,364). happensAt(active(lb1gt_2),364).
orientation(lb1gt_1,151,365). coords(lb1gt_1,166,170,365).  visible(lb1gt_1,365). happensAt(active(lb1gt_1),365). orientation(lb1gt_2,146,365). coords(lb1gt_2,190,143,365).  visible(lb1gt_2,365). happensAt(active(lb1gt_2),365).
orientation(lb1gt_1,151,366). coords(lb1gt_1,166,170,366).  visible(lb1gt_1,366). happensAt(active(lb1gt_1),366). orientation(lb1gt_2,146,366). coords(lb1gt_2,190,143,366).  visible(lb1gt_2,366). happensAt(active(lb1gt_2),366).
orientation(lb1gt_1,151,367). coords(lb1gt_1,166,170,367).  visible(lb1gt_1,367). happensAt(active(lb1gt_1),367). orientation(lb1gt_2,146,367). coords(lb1gt_2,190,143,367).  visible(lb1gt_2,367). happensAt(active(lb1gt_2),367).
orientation(lb1gt_1,151,368). coords(lb1gt_1,166,170,368).  visible(lb1gt_1,368). happensAt(active(lb1gt_1),368). orientation(lb1gt_2,146,368). coords(lb1gt_2,190,143,368).  visible(lb1gt_2,368). happensAt(active(lb1gt_2),368).
orientation(lb1gt_1,151,369). coords(lb1gt_1,166,170,369).  visible(lb1gt_1,369). happensAt(active(lb1gt_1),369). orientation(lb1gt_2,146,369). coords(lb1gt_2,191,143,369).  visible(lb1gt_2,369). happensAt(active(lb1gt_2),369).
orientation(lb1gt_1,151,370). coords(lb1gt_1,166,170,370).  visible(lb1gt_1,370). happensAt(active(lb1gt_1),370). orientation(lb1gt_2,146,370). coords(lb1gt_2,191,143,370).  visible(lb1gt_2,370). happensAt(active(lb1gt_2),370).
orientation(lb1gt_1,151,371). coords(lb1gt_1,166,170,371).  visible(lb1gt_1,371). happensAt(active(lb1gt_1),371). orientation(lb1gt_2,146,371). coords(lb1gt_2,191,143,371).  visible(lb1gt_2,371). happensAt(active(lb1gt_2),371).
orientation(lb1gt_1,151,372). coords(lb1gt_1,167,170,372).  visible(lb1gt_1,372). happensAt(active(lb1gt_1),372). orientation(lb1gt_2,146,372). coords(lb1gt_2,191,142,372).  visible(lb1gt_2,372). happensAt(active(lb1gt_2),372).
#show holdsAt/2.