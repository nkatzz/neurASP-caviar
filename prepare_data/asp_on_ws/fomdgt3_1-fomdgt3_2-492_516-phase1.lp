
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

	orientation(fomdgt3_1,150,492). coords(fomdgt3_1,46,102,492).  visible(fomdgt3_1,492). happensAt(inactive(fomdgt3_1),492). orientation(fomdgt3_2,57,492). coords(fomdgt3_2,249,101,492).  visible(fomdgt3_2,492). happensAt(walking(fomdgt3_2),492).
orientation(fomdgt3_1,150,493). coords(fomdgt3_1,46,102,493).  visible(fomdgt3_1,493). happensAt(inactive(fomdgt3_1),493). orientation(fomdgt3_2,57,493). coords(fomdgt3_2,249,101,493).  visible(fomdgt3_2,493). happensAt(walking(fomdgt3_2),493).
orientation(fomdgt3_1,150,494). coords(fomdgt3_1,46,102,494).  visible(fomdgt3_1,494). happensAt(inactive(fomdgt3_1),494). orientation(fomdgt3_2,57,494). coords(fomdgt3_2,249,101,494).  visible(fomdgt3_2,494). happensAt(walking(fomdgt3_2),494).
orientation(fomdgt3_1,150,495). coords(fomdgt3_1,46,102,495).  visible(fomdgt3_1,495). happensAt(inactive(fomdgt3_1),495). orientation(fomdgt3_2,71,495). coords(fomdgt3_2,249,100,495).  visible(fomdgt3_2,495). happensAt(walking(fomdgt3_2),495).
orientation(fomdgt3_1,150,496). coords(fomdgt3_1,46,102,496).  visible(fomdgt3_1,496). happensAt(inactive(fomdgt3_1),496). orientation(fomdgt3_2,71,496). coords(fomdgt3_2,249,100,496).  visible(fomdgt3_2,496). happensAt(walking(fomdgt3_2),496).
orientation(fomdgt3_1,150,497). coords(fomdgt3_1,46,102,497).  visible(fomdgt3_1,497). happensAt(inactive(fomdgt3_1),497). orientation(fomdgt3_2,71,497). coords(fomdgt3_2,249,100,497).  visible(fomdgt3_2,497). happensAt(walking(fomdgt3_2),497).
orientation(fomdgt3_1,150,498). coords(fomdgt3_1,46,102,498).  visible(fomdgt3_1,498). happensAt(inactive(fomdgt3_1),498). orientation(fomdgt3_2,71,498). coords(fomdgt3_2,249,100,498).  visible(fomdgt3_2,498). happensAt(walking(fomdgt3_2),498).
orientation(fomdgt3_1,150,499). coords(fomdgt3_1,46,102,499).  visible(fomdgt3_1,499). happensAt(inactive(fomdgt3_1),499). orientation(fomdgt3_2,71,499). coords(fomdgt3_2,245,100,499).  visible(fomdgt3_2,499). happensAt(walking(fomdgt3_2),499).
orientation(fomdgt3_1,150,500). coords(fomdgt3_1,46,102,500).  visible(fomdgt3_1,500). happensAt(inactive(fomdgt3_1),500). orientation(fomdgt3_2,71,500). coords(fomdgt3_2,244,100,500).  visible(fomdgt3_2,500). happensAt(walking(fomdgt3_2),500).
orientation(fomdgt3_1,150,501). coords(fomdgt3_1,46,102,501).  visible(fomdgt3_1,501). happensAt(inactive(fomdgt3_1),501). orientation(fomdgt3_2,71,501). coords(fomdgt3_2,244,100,501).  visible(fomdgt3_2,501). happensAt(walking(fomdgt3_2),501).
orientation(fomdgt3_1,150,502). coords(fomdgt3_1,46,102,502).  visible(fomdgt3_1,502). happensAt(inactive(fomdgt3_1),502). orientation(fomdgt3_2,80,502). coords(fomdgt3_2,242,99,502).  visible(fomdgt3_2,502). happensAt(walking(fomdgt3_2),502).
orientation(fomdgt3_1,150,503). coords(fomdgt3_1,46,102,503).  visible(fomdgt3_1,503). happensAt(inactive(fomdgt3_1),503). orientation(fomdgt3_2,80,503). coords(fomdgt3_2,243,99,503).  visible(fomdgt3_2,503). happensAt(walking(fomdgt3_2),503).
orientation(fomdgt3_1,150,504). coords(fomdgt3_1,46,102,504).  visible(fomdgt3_1,504). happensAt(inactive(fomdgt3_1),504). orientation(fomdgt3_2,80,504). coords(fomdgt3_2,243,99,504).  visible(fomdgt3_2,504). happensAt(walking(fomdgt3_2),504).
orientation(fomdgt3_1,150,505). coords(fomdgt3_1,46,102,505).  visible(fomdgt3_1,505). happensAt(inactive(fomdgt3_1),505). orientation(fomdgt3_2,80,505). coords(fomdgt3_2,242,99,505).  visible(fomdgt3_2,505). happensAt(walking(fomdgt3_2),505).
orientation(fomdgt3_1,150,506). coords(fomdgt3_1,46,102,506).  visible(fomdgt3_1,506). happensAt(inactive(fomdgt3_1),506). orientation(fomdgt3_2,80,506). coords(fomdgt3_2,241,99,506).  visible(fomdgt3_2,506). happensAt(walking(fomdgt3_2),506).
orientation(fomdgt3_1,150,507). coords(fomdgt3_1,46,102,507).  visible(fomdgt3_1,507). happensAt(inactive(fomdgt3_1),507). orientation(fomdgt3_2,80,507). coords(fomdgt3_2,241,99,507).  visible(fomdgt3_2,507). happensAt(walking(fomdgt3_2),507).
orientation(fomdgt3_1,150,508). coords(fomdgt3_1,46,102,508).  visible(fomdgt3_1,508). happensAt(inactive(fomdgt3_1),508). orientation(fomdgt3_2,80,508). coords(fomdgt3_2,241,97,508).  visible(fomdgt3_2,508). happensAt(walking(fomdgt3_2),508).
orientation(fomdgt3_1,150,509). coords(fomdgt3_1,46,102,509).  visible(fomdgt3_1,509). happensAt(inactive(fomdgt3_1),509). orientation(fomdgt3_2,80,509). coords(fomdgt3_2,241,97,509).  visible(fomdgt3_2,509). happensAt(walking(fomdgt3_2),509).
orientation(fomdgt3_1,150,510). coords(fomdgt3_1,46,102,510).  visible(fomdgt3_1,510). happensAt(inactive(fomdgt3_1),510). orientation(fomdgt3_2,80,510). coords(fomdgt3_2,241,98,510).  visible(fomdgt3_2,510). happensAt(walking(fomdgt3_2),510).
orientation(fomdgt3_1,150,511). coords(fomdgt3_1,46,102,511).  visible(fomdgt3_1,511). happensAt(inactive(fomdgt3_1),511). orientation(fomdgt3_2,80,511). coords(fomdgt3_2,240,99,511).  visible(fomdgt3_2,511). happensAt(walking(fomdgt3_2),511).
orientation(fomdgt3_1,150,512). coords(fomdgt3_1,46,102,512).  visible(fomdgt3_1,512). happensAt(inactive(fomdgt3_1),512). orientation(fomdgt3_2,80,512). coords(fomdgt3_2,240,100,512).  visible(fomdgt3_2,512). happensAt(walking(fomdgt3_2),512).
orientation(fomdgt3_1,150,513). coords(fomdgt3_1,46,102,513).  visible(fomdgt3_1,513). happensAt(inactive(fomdgt3_1),513). orientation(fomdgt3_2,80,513). coords(fomdgt3_2,240,100,513).  visible(fomdgt3_2,513). happensAt(walking(fomdgt3_2),513).
orientation(fomdgt3_1,150,514). coords(fomdgt3_1,46,102,514).  visible(fomdgt3_1,514). happensAt(inactive(fomdgt3_1),514). orientation(fomdgt3_2,80,514). coords(fomdgt3_2,239,100,514).  visible(fomdgt3_2,514). happensAt(walking(fomdgt3_2),514).
orientation(fomdgt3_1,150,515). coords(fomdgt3_1,46,102,515).  visible(fomdgt3_1,515). happensAt(inactive(fomdgt3_1),515). orientation(fomdgt3_2,80,515). coords(fomdgt3_2,239,100,515).  visible(fomdgt3_2,515). happensAt(walking(fomdgt3_2),515).
orientation(fomdgt3_1,150,516). coords(fomdgt3_1,46,102,516).  visible(fomdgt3_1,516). happensAt(inactive(fomdgt3_1),516). orientation(fomdgt3_2,80,516). coords(fomdgt3_2,238,101,516).  visible(fomdgt3_2,516). happensAt(walking(fomdgt3_2),516).
#show holdsAt/2.