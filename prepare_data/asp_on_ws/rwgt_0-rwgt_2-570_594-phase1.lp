
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

	orientation(rwgt_0,137,570). coords(rwgt_0,316,283,570).  visible(rwgt_0,570). happensAt(inactive(rwgt_0),570). orientation(rwgt_2,144,570). coords(rwgt_2,146,42,570).  visible(rwgt_2,570). happensAt(walking(rwgt_2),570).
orientation(rwgt_0,137,571). coords(rwgt_0,316,283,571).  visible(rwgt_0,571). happensAt(inactive(rwgt_0),571). orientation(rwgt_2,144,571). coords(rwgt_2,146,43,571).  visible(rwgt_2,571). happensAt(walking(rwgt_2),571).
orientation(rwgt_0,137,572). coords(rwgt_0,316,283,572).  visible(rwgt_0,572). happensAt(inactive(rwgt_0),572). orientation(rwgt_2,144,572). coords(rwgt_2,145,43,572).  visible(rwgt_2,572). happensAt(walking(rwgt_2),572).
orientation(rwgt_0,137,573). coords(rwgt_0,316,283,573).  visible(rwgt_0,573). happensAt(inactive(rwgt_0),573). orientation(rwgt_2,144,573). coords(rwgt_2,144,43,573).  visible(rwgt_2,573). happensAt(walking(rwgt_2),573).
orientation(rwgt_0,137,574). coords(rwgt_0,316,283,574).  visible(rwgt_0,574). happensAt(inactive(rwgt_0),574). orientation(rwgt_2,144,574). coords(rwgt_2,144,43,574).  visible(rwgt_2,574). happensAt(walking(rwgt_2),574).
orientation(rwgt_0,137,575). coords(rwgt_0,316,283,575).  visible(rwgt_0,575). happensAt(inactive(rwgt_0),575). orientation(rwgt_2,144,575). coords(rwgt_2,143,43,575).  visible(rwgt_2,575). happensAt(walking(rwgt_2),575).
orientation(rwgt_0,137,576). coords(rwgt_0,316,283,576).  visible(rwgt_0,576). happensAt(inactive(rwgt_0),576). orientation(rwgt_2,144,576). coords(rwgt_2,143,44,576).  visible(rwgt_2,576). happensAt(walking(rwgt_2),576).
orientation(rwgt_0,137,577). coords(rwgt_0,316,283,577).  visible(rwgt_0,577). happensAt(inactive(rwgt_0),577). orientation(rwgt_2,144,577). coords(rwgt_2,142,44,577).  visible(rwgt_2,577). happensAt(walking(rwgt_2),577).
orientation(rwgt_0,137,578). coords(rwgt_0,316,283,578).  visible(rwgt_0,578). happensAt(inactive(rwgt_0),578). orientation(rwgt_2,144,578). coords(rwgt_2,142,44,578).  visible(rwgt_2,578). happensAt(walking(rwgt_2),578).
orientation(rwgt_0,137,579). coords(rwgt_0,316,283,579).  visible(rwgt_0,579). happensAt(inactive(rwgt_0),579). orientation(rwgt_2,144,579). coords(rwgt_2,142,44,579).  visible(rwgt_2,579). happensAt(walking(rwgt_2),579).
orientation(rwgt_0,137,580). coords(rwgt_0,316,283,580).  visible(rwgt_0,580). happensAt(inactive(rwgt_0),580). orientation(rwgt_2,144,580). coords(rwgt_2,142,44,580).  visible(rwgt_2,580). happensAt(walking(rwgt_2),580).
orientation(rwgt_0,137,581). coords(rwgt_0,316,283,581).  visible(rwgt_0,581). happensAt(inactive(rwgt_0),581). orientation(rwgt_2,144,581). coords(rwgt_2,143,45,581).  visible(rwgt_2,581). happensAt(walking(rwgt_2),581).
orientation(rwgt_0,137,582). coords(rwgt_0,316,283,582).  visible(rwgt_0,582). happensAt(inactive(rwgt_0),582). orientation(rwgt_2,144,582). coords(rwgt_2,143,44,582).  visible(rwgt_2,582). happensAt(walking(rwgt_2),582).
orientation(rwgt_0,137,583). coords(rwgt_0,316,283,583).  visible(rwgt_0,583). happensAt(inactive(rwgt_0),583). orientation(rwgt_2,144,583). coords(rwgt_2,143,45,583).  visible(rwgt_2,583). happensAt(walking(rwgt_2),583).
orientation(rwgt_0,137,584). coords(rwgt_0,316,283,584).  visible(rwgt_0,584). happensAt(inactive(rwgt_0),584). orientation(rwgt_2,144,584). coords(rwgt_2,144,45,584).  visible(rwgt_2,584). happensAt(walking(rwgt_2),584).
orientation(rwgt_0,137,585). coords(rwgt_0,316,283,585).  visible(rwgt_0,585). happensAt(inactive(rwgt_0),585). orientation(rwgt_2,144,585). coords(rwgt_2,143,45,585).  visible(rwgt_2,585). happensAt(walking(rwgt_2),585).
orientation(rwgt_0,137,586). coords(rwgt_0,316,283,586).  visible(rwgt_0,586). happensAt(inactive(rwgt_0),586). orientation(rwgt_2,144,586). coords(rwgt_2,142,45,586).  visible(rwgt_2,586). happensAt(walking(rwgt_2),586).
orientation(rwgt_0,137,587). coords(rwgt_0,316,283,587).  visible(rwgt_0,587). happensAt(inactive(rwgt_0),587). orientation(rwgt_2,144,587). coords(rwgt_2,142,45,587).  visible(rwgt_2,587). happensAt(walking(rwgt_2),587).
orientation(rwgt_0,137,588). coords(rwgt_0,316,283,588).  visible(rwgt_0,588). happensAt(inactive(rwgt_0),588). orientation(rwgt_2,144,588). coords(rwgt_2,142,45,588).  visible(rwgt_2,588). happensAt(walking(rwgt_2),588).
orientation(rwgt_0,137,589). coords(rwgt_0,316,283,589).  visible(rwgt_0,589). happensAt(inactive(rwgt_0),589). orientation(rwgt_2,144,589). coords(rwgt_2,142,45,589).  visible(rwgt_2,589). happensAt(walking(rwgt_2),589).
orientation(rwgt_0,137,590). coords(rwgt_0,316,283,590).  visible(rwgt_0,590). happensAt(inactive(rwgt_0),590). orientation(rwgt_2,144,590). coords(rwgt_2,142,45,590).  visible(rwgt_2,590). happensAt(walking(rwgt_2),590).
orientation(rwgt_0,137,591). coords(rwgt_0,316,283,591).  visible(rwgt_0,591). happensAt(inactive(rwgt_0),591). orientation(rwgt_2,144,591). coords(rwgt_2,142,45,591).  visible(rwgt_2,591). happensAt(walking(rwgt_2),591).
orientation(rwgt_0,137,592). coords(rwgt_0,316,283,592).  visible(rwgt_0,592). happensAt(inactive(rwgt_0),592). orientation(rwgt_2,144,592). coords(rwgt_2,142,46,592).  visible(rwgt_2,592). happensAt(walking(rwgt_2),592).
orientation(rwgt_0,137,593). coords(rwgt_0,316,283,593).  visible(rwgt_0,593). happensAt(inactive(rwgt_0),593). orientation(rwgt_2,144,593). coords(rwgt_2,142,46,593).  visible(rwgt_2,593). happensAt(walking(rwgt_2),593).
orientation(rwgt_0,137,594). coords(rwgt_0,316,283,594).  visible(rwgt_0,594). happensAt(inactive(rwgt_0),594). orientation(rwgt_2,144,594). coords(rwgt_2,142,48,594).  visible(rwgt_2,594). happensAt(walking(rwgt_2),594).
#show holdsAt/2.