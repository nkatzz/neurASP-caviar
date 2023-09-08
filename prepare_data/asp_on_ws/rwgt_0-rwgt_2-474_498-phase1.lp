
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

	orientation(rwgt_0,137,474). coords(rwgt_0,316,283,474).  visible(rwgt_0,474). happensAt(inactive(rwgt_0),474). orientation(rwgt_2,138,474). coords(rwgt_2,168,12,474).  happensAt(appear(rwgt_2),474). happensAt(walking(rwgt_2),474).
orientation(rwgt_0,137,475). coords(rwgt_0,316,283,475).  visible(rwgt_0,475). happensAt(inactive(rwgt_0),475). orientation(rwgt_2,132,475). coords(rwgt_2,169,13,475).  visible(rwgt_2,475). happensAt(walking(rwgt_2),475).
orientation(rwgt_0,137,476). coords(rwgt_0,316,283,476).  visible(rwgt_0,476). happensAt(inactive(rwgt_0),476). orientation(rwgt_2,135,476). coords(rwgt_2,168,16,476).  visible(rwgt_2,476). happensAt(walking(rwgt_2),476).
orientation(rwgt_0,137,477). coords(rwgt_0,316,283,477).  visible(rwgt_0,477). happensAt(inactive(rwgt_0),477). orientation(rwgt_2,135,477). coords(rwgt_2,169,17,477).  visible(rwgt_2,477). happensAt(walking(rwgt_2),477).
orientation(rwgt_0,137,478). coords(rwgt_0,316,283,478).  visible(rwgt_0,478). happensAt(inactive(rwgt_0),478). orientation(rwgt_2,135,478). coords(rwgt_2,168,17,478).  visible(rwgt_2,478). happensAt(walking(rwgt_2),478).
orientation(rwgt_0,137,479). coords(rwgt_0,316,283,479).  visible(rwgt_0,479). happensAt(inactive(rwgt_0),479). orientation(rwgt_2,135,479). coords(rwgt_2,168,19,479).  visible(rwgt_2,479). happensAt(walking(rwgt_2),479).
orientation(rwgt_0,137,480). coords(rwgt_0,316,283,480).  visible(rwgt_0,480). happensAt(inactive(rwgt_0),480). orientation(rwgt_2,128,480). coords(rwgt_2,168,19,480).  visible(rwgt_2,480). happensAt(walking(rwgt_2),480).
orientation(rwgt_0,137,481). coords(rwgt_0,316,283,481).  visible(rwgt_0,481). happensAt(inactive(rwgt_0),481). orientation(rwgt_2,128,481). coords(rwgt_2,168,19,481).  visible(rwgt_2,481). happensAt(walking(rwgt_2),481).
orientation(rwgt_0,137,482). coords(rwgt_0,316,283,482).  visible(rwgt_0,482). happensAt(inactive(rwgt_0),482). orientation(rwgt_2,128,482). coords(rwgt_2,168,18,482).  visible(rwgt_2,482). happensAt(walking(rwgt_2),482).
orientation(rwgt_0,137,483). coords(rwgt_0,316,283,483).  visible(rwgt_0,483). happensAt(inactive(rwgt_0),483). orientation(rwgt_2,125,483). coords(rwgt_2,168,19,483).  visible(rwgt_2,483). happensAt(walking(rwgt_2),483).
orientation(rwgt_0,137,484). coords(rwgt_0,316,283,484).  visible(rwgt_0,484). happensAt(inactive(rwgt_0),484). orientation(rwgt_2,125,484). coords(rwgt_2,168,19,484).  visible(rwgt_2,484). happensAt(walking(rwgt_2),484).
orientation(rwgt_0,137,485). coords(rwgt_0,316,283,485).  visible(rwgt_0,485). happensAt(inactive(rwgt_0),485). orientation(rwgt_2,125,485). coords(rwgt_2,168,19,485).  visible(rwgt_2,485). happensAt(walking(rwgt_2),485).
orientation(rwgt_0,137,486). coords(rwgt_0,316,283,486).  visible(rwgt_0,486). happensAt(inactive(rwgt_0),486). orientation(rwgt_2,125,486). coords(rwgt_2,168,19,486).  visible(rwgt_2,486). happensAt(walking(rwgt_2),486).
orientation(rwgt_0,137,487). coords(rwgt_0,316,283,487).  visible(rwgt_0,487). happensAt(inactive(rwgt_0),487). orientation(rwgt_2,125,487). coords(rwgt_2,168,19,487).  visible(rwgt_2,487). happensAt(walking(rwgt_2),487).
orientation(rwgt_0,137,488). coords(rwgt_0,316,283,488).  visible(rwgt_0,488). happensAt(inactive(rwgt_0),488). orientation(rwgt_2,125,488). coords(rwgt_2,168,20,488).  visible(rwgt_2,488). happensAt(walking(rwgt_2),488).
orientation(rwgt_0,137,489). coords(rwgt_0,316,283,489).  visible(rwgt_0,489). happensAt(inactive(rwgt_0),489). orientation(rwgt_2,124,489). coords(rwgt_2,167,20,489).  visible(rwgt_2,489). happensAt(walking(rwgt_2),489).
orientation(rwgt_0,137,490). coords(rwgt_0,316,283,490).  visible(rwgt_0,490). happensAt(inactive(rwgt_0),490). orientation(rwgt_2,124,490). coords(rwgt_2,167,20,490).  visible(rwgt_2,490). happensAt(walking(rwgt_2),490).
orientation(rwgt_0,137,491). coords(rwgt_0,316,283,491).  visible(rwgt_0,491). happensAt(inactive(rwgt_0),491). orientation(rwgt_2,124,491). coords(rwgt_2,167,21,491).  visible(rwgt_2,491). happensAt(walking(rwgt_2),491).
orientation(rwgt_0,137,492). coords(rwgt_0,316,283,492).  visible(rwgt_0,492). happensAt(inactive(rwgt_0),492). orientation(rwgt_2,127,492). coords(rwgt_2,167,21,492).  visible(rwgt_2,492). happensAt(walking(rwgt_2),492).
orientation(rwgt_0,137,493). coords(rwgt_0,316,283,493).  visible(rwgt_0,493). happensAt(inactive(rwgt_0),493). orientation(rwgt_2,124,493). coords(rwgt_2,167,22,493).  visible(rwgt_2,493). happensAt(walking(rwgt_2),493).
orientation(rwgt_0,137,494). coords(rwgt_0,316,283,494).  visible(rwgt_0,494). happensAt(inactive(rwgt_0),494). orientation(rwgt_2,124,494). coords(rwgt_2,167,22,494).  visible(rwgt_2,494). happensAt(walking(rwgt_2),494).
orientation(rwgt_0,137,495). coords(rwgt_0,316,283,495).  visible(rwgt_0,495). happensAt(inactive(rwgt_0),495). orientation(rwgt_2,124,495). coords(rwgt_2,167,22,495).  visible(rwgt_2,495). happensAt(walking(rwgt_2),495).
orientation(rwgt_0,137,496). coords(rwgt_0,316,283,496).  visible(rwgt_0,496). happensAt(inactive(rwgt_0),496). orientation(rwgt_2,124,496). coords(rwgt_2,166,24,496).  visible(rwgt_2,496). happensAt(walking(rwgt_2),496).
orientation(rwgt_0,137,497). coords(rwgt_0,316,283,497).  visible(rwgt_0,497). happensAt(inactive(rwgt_0),497). orientation(rwgt_2,124,497). coords(rwgt_2,166,26,497).  visible(rwgt_2,497). happensAt(walking(rwgt_2),497).
orientation(rwgt_0,137,498). coords(rwgt_0,316,283,498).  visible(rwgt_0,498). happensAt(inactive(rwgt_0),498). orientation(rwgt_2,124,498). coords(rwgt_2,166,27,498).  visible(rwgt_2,498). happensAt(walking(rwgt_2),498).
#show holdsAt/2.