
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

	orientation(fomdgt3_1,150,482). coords(fomdgt3_1,46,102,482).  visible(fomdgt3_1,482). happensAt(inactive(fomdgt3_1),482). orientation(fomdgt3_7,159,482). coords(fomdgt3_7,113,153,482).  visible(fomdgt3_7,482). happensAt(walking(fomdgt3_7),482).
orientation(fomdgt3_1,150,483). coords(fomdgt3_1,46,102,483).  visible(fomdgt3_1,483). happensAt(inactive(fomdgt3_1),483). orientation(fomdgt3_7,159,483). coords(fomdgt3_7,113,154,483).  visible(fomdgt3_7,483). happensAt(walking(fomdgt3_7),483).
orientation(fomdgt3_1,150,484). coords(fomdgt3_1,46,102,484).  visible(fomdgt3_1,484). happensAt(inactive(fomdgt3_1),484). orientation(fomdgt3_7,159,484). coords(fomdgt3_7,116,156,484).  visible(fomdgt3_7,484). happensAt(walking(fomdgt3_7),484).
orientation(fomdgt3_1,150,485). coords(fomdgt3_1,46,102,485).  visible(fomdgt3_1,485). happensAt(inactive(fomdgt3_1),485). orientation(fomdgt3_7,159,485). coords(fomdgt3_7,116,157,485).  visible(fomdgt3_7,485). happensAt(walking(fomdgt3_7),485).
orientation(fomdgt3_1,150,486). coords(fomdgt3_1,46,102,486).  visible(fomdgt3_1,486). happensAt(inactive(fomdgt3_1),486). orientation(fomdgt3_7,159,486). coords(fomdgt3_7,117,157,486).  visible(fomdgt3_7,486). happensAt(walking(fomdgt3_7),486).
orientation(fomdgt3_1,150,487). coords(fomdgt3_1,46,102,487).  visible(fomdgt3_1,487). happensAt(inactive(fomdgt3_1),487). orientation(fomdgt3_7,159,487). coords(fomdgt3_7,118,158,487).  visible(fomdgt3_7,487). happensAt(walking(fomdgt3_7),487).
orientation(fomdgt3_1,150,488). coords(fomdgt3_1,46,102,488).  visible(fomdgt3_1,488). happensAt(inactive(fomdgt3_1),488). orientation(fomdgt3_7,159,488). coords(fomdgt3_7,118,160,488).  visible(fomdgt3_7,488). happensAt(walking(fomdgt3_7),488).
orientation(fomdgt3_1,150,489). coords(fomdgt3_1,46,102,489).  visible(fomdgt3_1,489). happensAt(inactive(fomdgt3_1),489). orientation(fomdgt3_7,159,489). coords(fomdgt3_7,118,161,489).  visible(fomdgt3_7,489). happensAt(walking(fomdgt3_7),489).
orientation(fomdgt3_1,150,490). coords(fomdgt3_1,46,102,490).  visible(fomdgt3_1,490). happensAt(inactive(fomdgt3_1),490). orientation(fomdgt3_7,159,490). coords(fomdgt3_7,119,162,490).  visible(fomdgt3_7,490). happensAt(walking(fomdgt3_7),490).
orientation(fomdgt3_1,150,491). coords(fomdgt3_1,46,102,491).  visible(fomdgt3_1,491). happensAt(inactive(fomdgt3_1),491). orientation(fomdgt3_7,159,491). coords(fomdgt3_7,119,163,491).  visible(fomdgt3_7,491). happensAt(walking(fomdgt3_7),491).
orientation(fomdgt3_1,150,492). coords(fomdgt3_1,46,102,492).  visible(fomdgt3_1,492). happensAt(inactive(fomdgt3_1),492). orientation(fomdgt3_7,159,492). coords(fomdgt3_7,119,163,492).  visible(fomdgt3_7,492). happensAt(walking(fomdgt3_7),492).
orientation(fomdgt3_1,150,493). coords(fomdgt3_1,46,102,493).  visible(fomdgt3_1,493). happensAt(inactive(fomdgt3_1),493). orientation(fomdgt3_7,159,493). coords(fomdgt3_7,119,164,493).  visible(fomdgt3_7,493). happensAt(walking(fomdgt3_7),493).
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
#show holdsAt/2.