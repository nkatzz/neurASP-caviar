
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

	orientation(fomdgt3_1,150,494). coords(fomdgt3_1,46,102,494).  visible(fomdgt3_1,494). happensAt(inactive(fomdgt3_1),494). orientation(fomdgt3_7,159,494). coords(fomdgt3_7,119,165,494).  visible(fomdgt3_7,494). happensAt(walking(fomdgt3_7),494).
orientation(fomdgt3_1,150,495). coords(fomdgt3_1,46,102,495).  visible(fomdgt3_1,495). happensAt(inactive(fomdgt3_1),495). orientation(fomdgt3_7,159,495). coords(fomdgt3_7,119,166,495).  visible(fomdgt3_7,495). happensAt(walking(fomdgt3_7),495).
orientation(fomdgt3_1,150,496). coords(fomdgt3_1,46,102,496).  visible(fomdgt3_1,496). happensAt(inactive(fomdgt3_1),496). orientation(fomdgt3_7,159,496). coords(fomdgt3_7,119,168,496).  visible(fomdgt3_7,496). happensAt(walking(fomdgt3_7),496).
orientation(fomdgt3_1,150,497). coords(fomdgt3_1,46,102,497).  visible(fomdgt3_1,497). happensAt(inactive(fomdgt3_1),497). orientation(fomdgt3_7,159,497). coords(fomdgt3_7,121,169,497).  visible(fomdgt3_7,497). happensAt(walking(fomdgt3_7),497).
orientation(fomdgt3_1,150,498). coords(fomdgt3_1,46,102,498).  visible(fomdgt3_1,498). happensAt(inactive(fomdgt3_1),498). orientation(fomdgt3_7,159,498). coords(fomdgt3_7,121,169,498).  visible(fomdgt3_7,498). happensAt(walking(fomdgt3_7),498).
orientation(fomdgt3_1,150,499). coords(fomdgt3_1,46,102,499).  visible(fomdgt3_1,499). happensAt(inactive(fomdgt3_1),499). orientation(fomdgt3_7,159,499). coords(fomdgt3_7,123,174,499).  visible(fomdgt3_7,499). happensAt(walking(fomdgt3_7),499).
orientation(fomdgt3_1,150,500). coords(fomdgt3_1,46,102,500).  visible(fomdgt3_1,500). happensAt(inactive(fomdgt3_1),500). orientation(fomdgt3_7,159,500). coords(fomdgt3_7,123,175,500).  visible(fomdgt3_7,500). happensAt(walking(fomdgt3_7),500).
orientation(fomdgt3_1,150,501). coords(fomdgt3_1,46,102,501).  visible(fomdgt3_1,501). happensAt(inactive(fomdgt3_1),501). orientation(fomdgt3_7,159,501). coords(fomdgt3_7,124,175,501).  visible(fomdgt3_7,501). happensAt(walking(fomdgt3_7),501).
orientation(fomdgt3_1,150,502). coords(fomdgt3_1,46,102,502).  visible(fomdgt3_1,502). happensAt(inactive(fomdgt3_1),502). orientation(fomdgt3_7,159,502). coords(fomdgt3_7,126,177,502).  visible(fomdgt3_7,502). happensAt(walking(fomdgt3_7),502).
orientation(fomdgt3_1,150,503). coords(fomdgt3_1,46,102,503).  visible(fomdgt3_1,503). happensAt(inactive(fomdgt3_1),503). orientation(fomdgt3_7,159,503). coords(fomdgt3_7,128,177,503).  visible(fomdgt3_7,503). happensAt(walking(fomdgt3_7),503).
orientation(fomdgt3_1,150,504). coords(fomdgt3_1,46,102,504).  visible(fomdgt3_1,504). happensAt(inactive(fomdgt3_1),504). orientation(fomdgt3_7,159,504). coords(fomdgt3_7,128,182,504).  visible(fomdgt3_7,504). happensAt(walking(fomdgt3_7),504).
orientation(fomdgt3_1,150,505). coords(fomdgt3_1,46,102,505).  visible(fomdgt3_1,505). happensAt(inactive(fomdgt3_1),505). orientation(fomdgt3_7,159,505). coords(fomdgt3_7,128,183,505).  visible(fomdgt3_7,505). happensAt(walking(fomdgt3_7),505).
orientation(fomdgt3_1,150,506). coords(fomdgt3_1,46,102,506).  visible(fomdgt3_1,506). happensAt(inactive(fomdgt3_1),506). orientation(fomdgt3_7,159,506). coords(fomdgt3_7,129,184,506).  visible(fomdgt3_7,506). happensAt(walking(fomdgt3_7),506).
orientation(fomdgt3_1,150,507). coords(fomdgt3_1,46,102,507).  visible(fomdgt3_1,507). happensAt(inactive(fomdgt3_1),507). orientation(fomdgt3_7,159,507). coords(fomdgt3_7,130,184,507).  visible(fomdgt3_7,507). happensAt(walking(fomdgt3_7),507).
orientation(fomdgt3_1,150,508). coords(fomdgt3_1,46,102,508).  visible(fomdgt3_1,508). happensAt(inactive(fomdgt3_1),508). orientation(fomdgt3_7,159,508). coords(fomdgt3_7,130,184,508).  visible(fomdgt3_7,508). happensAt(walking(fomdgt3_7),508).
orientation(fomdgt3_1,150,509). coords(fomdgt3_1,46,102,509).  visible(fomdgt3_1,509). happensAt(inactive(fomdgt3_1),509). orientation(fomdgt3_7,159,509). coords(fomdgt3_7,130,185,509).  visible(fomdgt3_7,509). happensAt(walking(fomdgt3_7),509).
orientation(fomdgt3_1,150,510). coords(fomdgt3_1,46,102,510).  visible(fomdgt3_1,510). happensAt(inactive(fomdgt3_1),510). orientation(fomdgt3_7,159,510). coords(fomdgt3_7,131,185,510).  visible(fomdgt3_7,510). happensAt(walking(fomdgt3_7),510).
orientation(fomdgt3_1,150,511). coords(fomdgt3_1,46,102,511).  visible(fomdgt3_1,511). happensAt(inactive(fomdgt3_1),511). orientation(fomdgt3_7,159,511). coords(fomdgt3_7,131,186,511).  visible(fomdgt3_7,511). happensAt(walking(fomdgt3_7),511).
orientation(fomdgt3_1,150,512). coords(fomdgt3_1,46,102,512).  visible(fomdgt3_1,512). happensAt(inactive(fomdgt3_1),512). orientation(fomdgt3_7,159,512). coords(fomdgt3_7,131,189,512).  visible(fomdgt3_7,512). happensAt(walking(fomdgt3_7),512).
orientation(fomdgt3_1,150,513). coords(fomdgt3_1,46,102,513).  visible(fomdgt3_1,513). happensAt(inactive(fomdgt3_1),513). orientation(fomdgt3_7,159,513). coords(fomdgt3_7,131,191,513).  visible(fomdgt3_7,513). happensAt(walking(fomdgt3_7),513).
orientation(fomdgt3_1,150,514). coords(fomdgt3_1,46,102,514).  visible(fomdgt3_1,514). happensAt(inactive(fomdgt3_1),514). orientation(fomdgt3_7,159,514). coords(fomdgt3_7,138,191,514).  visible(fomdgt3_7,514). happensAt(walking(fomdgt3_7),514).
orientation(fomdgt3_1,150,515). coords(fomdgt3_1,46,102,515).  visible(fomdgt3_1,515). happensAt(inactive(fomdgt3_1),515). orientation(fomdgt3_7,159,515). coords(fomdgt3_7,138,191,515).  visible(fomdgt3_7,515). happensAt(walking(fomdgt3_7),515).
orientation(fomdgt3_1,150,516). coords(fomdgt3_1,46,102,516).  visible(fomdgt3_1,516). happensAt(inactive(fomdgt3_1),516). orientation(fomdgt3_7,159,516). coords(fomdgt3_7,141,194,516).  visible(fomdgt3_7,516). happensAt(walking(fomdgt3_7),516).
orientation(fomdgt3_1,150,517). coords(fomdgt3_1,46,102,517).  visible(fomdgt3_1,517). happensAt(inactive(fomdgt3_1),517). orientation(fomdgt3_7,159,517). coords(fomdgt3_7,141,195,517).  visible(fomdgt3_7,517). happensAt(walking(fomdgt3_7),517).
orientation(fomdgt3_1,150,518). coords(fomdgt3_1,46,102,518).  visible(fomdgt3_1,518). happensAt(inactive(fomdgt3_1),518). orientation(fomdgt3_7,159,518). coords(fomdgt3_7,145,195,518).  visible(fomdgt3_7,518). happensAt(walking(fomdgt3_7),518).
#show holdsAt/2.