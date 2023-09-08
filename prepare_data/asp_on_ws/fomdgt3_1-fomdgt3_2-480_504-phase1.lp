
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

	orientation(fomdgt3_1,150,480). coords(fomdgt3_1,46,102,480).  visible(fomdgt3_1,480). happensAt(inactive(fomdgt3_1),480). orientation(fomdgt3_2,42,480). coords(fomdgt3_2,249,98,480).  visible(fomdgt3_2,480). happensAt(active(fomdgt3_2),480).
orientation(fomdgt3_1,150,481). coords(fomdgt3_1,46,102,481).  visible(fomdgt3_1,481). happensAt(inactive(fomdgt3_1),481). orientation(fomdgt3_2,42,481). coords(fomdgt3_2,250,98,481).  visible(fomdgt3_2,481). happensAt(active(fomdgt3_2),481).
orientation(fomdgt3_1,150,482). coords(fomdgt3_1,46,102,482).  visible(fomdgt3_1,482). happensAt(inactive(fomdgt3_1),482). orientation(fomdgt3_2,42,482). coords(fomdgt3_2,250,99,482).  visible(fomdgt3_2,482). happensAt(active(fomdgt3_2),482).
orientation(fomdgt3_1,150,483). coords(fomdgt3_1,46,102,483).  visible(fomdgt3_1,483). happensAt(inactive(fomdgt3_1),483). orientation(fomdgt3_2,42,483). coords(fomdgt3_2,250,99,483).  visible(fomdgt3_2,483). happensAt(active(fomdgt3_2),483).
orientation(fomdgt3_1,150,484). coords(fomdgt3_1,46,102,484).  visible(fomdgt3_1,484). happensAt(inactive(fomdgt3_1),484). orientation(fomdgt3_2,55,484). coords(fomdgt3_2,250,99,484).  visible(fomdgt3_2,484). happensAt(active(fomdgt3_2),484).
orientation(fomdgt3_1,150,485). coords(fomdgt3_1,46,102,485).  visible(fomdgt3_1,485). happensAt(inactive(fomdgt3_1),485). orientation(fomdgt3_2,57,485). coords(fomdgt3_2,250,99,485).  visible(fomdgt3_2,485). happensAt(active(fomdgt3_2),485).
orientation(fomdgt3_1,150,486). coords(fomdgt3_1,46,102,486).  visible(fomdgt3_1,486). happensAt(inactive(fomdgt3_1),486). orientation(fomdgt3_2,57,486). coords(fomdgt3_2,251,100,486).  visible(fomdgt3_2,486). happensAt(active(fomdgt3_2),486).
orientation(fomdgt3_1,150,487). coords(fomdgt3_1,46,102,487).  visible(fomdgt3_1,487). happensAt(inactive(fomdgt3_1),487). orientation(fomdgt3_2,57,487). coords(fomdgt3_2,251,100,487).  visible(fomdgt3_2,487). happensAt(active(fomdgt3_2),487).
orientation(fomdgt3_1,150,488). coords(fomdgt3_1,46,102,488).  visible(fomdgt3_1,488). happensAt(inactive(fomdgt3_1),488). orientation(fomdgt3_2,57,488). coords(fomdgt3_2,251,100,488).  visible(fomdgt3_2,488). happensAt(active(fomdgt3_2),488).
orientation(fomdgt3_1,150,489). coords(fomdgt3_1,46,102,489).  visible(fomdgt3_1,489). happensAt(inactive(fomdgt3_1),489). orientation(fomdgt3_2,57,489). coords(fomdgt3_2,250,101,489).  visible(fomdgt3_2,489). happensAt(active(fomdgt3_2),489).
orientation(fomdgt3_1,150,490). coords(fomdgt3_1,46,102,490).  visible(fomdgt3_1,490). happensAt(inactive(fomdgt3_1),490). orientation(fomdgt3_2,57,490). coords(fomdgt3_2,250,102,490).  visible(fomdgt3_2,490). happensAt(active(fomdgt3_2),490).
orientation(fomdgt3_1,150,491). coords(fomdgt3_1,46,102,491).  visible(fomdgt3_1,491). happensAt(inactive(fomdgt3_1),491). orientation(fomdgt3_2,57,491). coords(fomdgt3_2,249,101,491).  visible(fomdgt3_2,491). happensAt(active(fomdgt3_2),491).
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
#show holdsAt/2.