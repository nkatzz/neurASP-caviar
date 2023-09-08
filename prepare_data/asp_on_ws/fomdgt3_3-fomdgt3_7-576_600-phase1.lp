
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

	orientation(fomdgt3_3,159,576). coords(fomdgt3_3,50,101,576).  visible(fomdgt3_3,576). happensAt(walking(fomdgt3_3),576). orientation(fomdgt3_7,171,576). coords(fomdgt3_7,222,260,576).  visible(fomdgt3_7,576). happensAt(walking(fomdgt3_7),576).
orientation(fomdgt3_3,159,577). coords(fomdgt3_3,50,101,577).  visible(fomdgt3_3,577). happensAt(walking(fomdgt3_3),577). orientation(fomdgt3_7,171,577). coords(fomdgt3_7,225,261,577).  visible(fomdgt3_7,577). happensAt(walking(fomdgt3_7),577).
orientation(fomdgt3_3,159,578). coords(fomdgt3_3,50,101,578).  visible(fomdgt3_3,578). happensAt(walking(fomdgt3_3),578). orientation(fomdgt3_7,171,578). coords(fomdgt3_7,225,262,578).  visible(fomdgt3_7,578). happensAt(walking(fomdgt3_7),578).
orientation(fomdgt3_3,159,579). coords(fomdgt3_3,50,101,579).  visible(fomdgt3_3,579). happensAt(walking(fomdgt3_3),579). orientation(fomdgt3_7,171,579). coords(fomdgt3_7,227,263,579).  visible(fomdgt3_7,579). happensAt(walking(fomdgt3_7),579).
orientation(fomdgt3_3,159,580). coords(fomdgt3_3,50,101,580).  visible(fomdgt3_3,580). happensAt(walking(fomdgt3_3),580). orientation(fomdgt3_7,171,580). coords(fomdgt3_7,227,263,580).  visible(fomdgt3_7,580). happensAt(walking(fomdgt3_7),580).
orientation(fomdgt3_3,159,581). coords(fomdgt3_3,50,101,581).  visible(fomdgt3_3,581). happensAt(walking(fomdgt3_3),581). orientation(fomdgt3_7,20,581). coords(fomdgt3_7,227,264,581).  visible(fomdgt3_7,581). happensAt(walking(fomdgt3_7),581).
orientation(fomdgt3_3,159,582). coords(fomdgt3_3,50,101,582).  visible(fomdgt3_3,582). happensAt(walking(fomdgt3_3),582). orientation(fomdgt3_7,20,582). coords(fomdgt3_7,227,264,582).  visible(fomdgt3_7,582). happensAt(walking(fomdgt3_7),582).
orientation(fomdgt3_3,159,583). coords(fomdgt3_3,52,100,583).  visible(fomdgt3_3,583). happensAt(walking(fomdgt3_3),583). orientation(fomdgt3_7,20,583). coords(fomdgt3_7,227,266,583).  visible(fomdgt3_7,583). happensAt(walking(fomdgt3_7),583).
orientation(fomdgt3_3,159,584). coords(fomdgt3_3,52,100,584).  visible(fomdgt3_3,584). happensAt(walking(fomdgt3_3),584). orientation(fomdgt3_7,20,584). coords(fomdgt3_7,229,268,584).  visible(fomdgt3_7,584). happensAt(walking(fomdgt3_7),584).
orientation(fomdgt3_3,159,585). coords(fomdgt3_3,52,100,585).  visible(fomdgt3_3,585). happensAt(walking(fomdgt3_3),585). orientation(fomdgt3_7,30,585). coords(fomdgt3_7,229,270,585).  visible(fomdgt3_7,585). happensAt(walking(fomdgt3_7),585).
orientation(fomdgt3_3,159,586). coords(fomdgt3_3,52,100,586).  visible(fomdgt3_3,586). happensAt(walking(fomdgt3_3),586). orientation(fomdgt3_7,30,586). coords(fomdgt3_7,229,272,586).  visible(fomdgt3_7,586). happensAt(walking(fomdgt3_7),586).
orientation(fomdgt3_3,159,587). coords(fomdgt3_3,52,100,587).  visible(fomdgt3_3,587). happensAt(walking(fomdgt3_3),587). orientation(fomdgt3_7,30,587). coords(fomdgt3_7,229,274,587).  visible(fomdgt3_7,587). happensAt(walking(fomdgt3_7),587).
orientation(fomdgt3_3,159,588). coords(fomdgt3_3,53,100,588).  visible(fomdgt3_3,588). happensAt(walking(fomdgt3_3),588). orientation(fomdgt3_7,30,588). coords(fomdgt3_7,231,274,588).  visible(fomdgt3_7,588). happensAt(walking(fomdgt3_7),588).
orientation(fomdgt3_3,159,589). coords(fomdgt3_3,55,99,589).  visible(fomdgt3_3,589). happensAt(walking(fomdgt3_3),589). orientation(fomdgt3_7,30,589). coords(fomdgt3_7,231,276,589).  visible(fomdgt3_7,589). happensAt(walking(fomdgt3_7),589).
orientation(fomdgt3_3,159,590). coords(fomdgt3_3,55,99,590).  visible(fomdgt3_3,590). happensAt(walking(fomdgt3_3),590). orientation(fomdgt3_7,30,590). coords(fomdgt3_7,231,276,590).  visible(fomdgt3_7,590). happensAt(walking(fomdgt3_7),590).
orientation(fomdgt3_3,159,591). coords(fomdgt3_3,57,99,591).  visible(fomdgt3_3,591). happensAt(walking(fomdgt3_3),591). orientation(fomdgt3_7,30,591). coords(fomdgt3_7,231,276,591).  visible(fomdgt3_7,591). happensAt(walking(fomdgt3_7),591).
orientation(fomdgt3_3,159,592). coords(fomdgt3_3,57,100,592).  visible(fomdgt3_3,592). happensAt(walking(fomdgt3_3),592). orientation(fomdgt3_7,30,592). coords(fomdgt3_7,231,276,592).  visible(fomdgt3_7,592). happensAt(walking(fomdgt3_7),592).
orientation(fomdgt3_3,159,593). coords(fomdgt3_3,57,99,593).  visible(fomdgt3_3,593). happensAt(walking(fomdgt3_3),593). orientation(fomdgt3_7,30,593). coords(fomdgt3_7,231,276,593).  visible(fomdgt3_7,593). happensAt(walking(fomdgt3_7),593).
orientation(fomdgt3_3,159,594). coords(fomdgt3_3,57,98,594).  visible(fomdgt3_3,594). happensAt(walking(fomdgt3_3),594). orientation(fomdgt3_7,30,594). coords(fomdgt3_7,234,276,594).  visible(fomdgt3_7,594). happensAt(walking(fomdgt3_7),594).
orientation(fomdgt3_3,159,595). coords(fomdgt3_3,57,98,595).  visible(fomdgt3_3,595). happensAt(walking(fomdgt3_3),595). orientation(fomdgt3_7,70,595). coords(fomdgt3_7,244,276,595).  visible(fomdgt3_7,595). happensAt(walking(fomdgt3_7),595).
orientation(fomdgt3_3,159,596). coords(fomdgt3_3,58,98,596).  visible(fomdgt3_3,596). happensAt(walking(fomdgt3_3),596). orientation(fomdgt3_7,70,596). coords(fomdgt3_7,244,277,596).  visible(fomdgt3_7,596). happensAt(walking(fomdgt3_7),596).
orientation(fomdgt3_3,159,597). coords(fomdgt3_3,59,98,597).  visible(fomdgt3_3,597). happensAt(walking(fomdgt3_3),597). orientation(fomdgt3_7,70,597). coords(fomdgt3_7,244,278,597).  visible(fomdgt3_7,597). happensAt(walking(fomdgt3_7),597).
orientation(fomdgt3_3,159,598). coords(fomdgt3_3,59,98,598).  visible(fomdgt3_3,598). happensAt(walking(fomdgt3_3),598). orientation(fomdgt3_7,70,598). coords(fomdgt3_7,244,279,598).  visible(fomdgt3_7,598). happensAt(walking(fomdgt3_7),598).
orientation(fomdgt3_3,159,599). coords(fomdgt3_3,60,98,599).  visible(fomdgt3_3,599). happensAt(walking(fomdgt3_3),599). orientation(fomdgt3_7,70,599). coords(fomdgt3_7,244,280,599).  visible(fomdgt3_7,599). happensAt(walking(fomdgt3_7),599).
orientation(fomdgt3_3,159,600). coords(fomdgt3_3,60,98,600).  visible(fomdgt3_3,600). happensAt(walking(fomdgt3_3),600). orientation(fomdgt3_7,70,600). coords(fomdgt3_7,244,280,600).  happensAt(disappear(fomdgt3_7),600). happensAt(walking(fomdgt3_7),600).
#show holdsAt/2.