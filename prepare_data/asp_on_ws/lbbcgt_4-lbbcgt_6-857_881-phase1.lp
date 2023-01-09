
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

	orientation(lbbcgt_4,131,857). coords(lbbcgt_4,5,83,857).  visible(lbbcgt_4,857). happensAt(walking(lbbcgt_4),857). orientation(lbbcgt_6,130,857). coords(lbbcgt_6,171,31,857).  visible(lbbcgt_6,857). happensAt(running(lbbcgt_6),857).
orientation(lbbcgt_4,131,858). coords(lbbcgt_4,6,81,858).  visible(lbbcgt_4,858). happensAt(walking(lbbcgt_4),858). orientation(lbbcgt_6,130,858). coords(lbbcgt_6,170,32,858).  visible(lbbcgt_6,858). happensAt(running(lbbcgt_6),858).
orientation(lbbcgt_4,131,859). coords(lbbcgt_4,6,79,859).  visible(lbbcgt_4,859). happensAt(walking(lbbcgt_4),859). orientation(lbbcgt_6,130,859). coords(lbbcgt_6,169,32,859).  visible(lbbcgt_6,859). happensAt(running(lbbcgt_6),859).
orientation(lbbcgt_4,131,860). coords(lbbcgt_4,6,79,860).  visible(lbbcgt_4,860). happensAt(walking(lbbcgt_4),860). orientation(lbbcgt_6,130,860). coords(lbbcgt_6,168,33,860).  visible(lbbcgt_6,860). happensAt(running(lbbcgt_6),860).
orientation(lbbcgt_4,131,861). coords(lbbcgt_4,6,80,861).  visible(lbbcgt_4,861). happensAt(walking(lbbcgt_4),861). orientation(lbbcgt_6,130,861). coords(lbbcgt_6,167,35,861).  visible(lbbcgt_6,861). happensAt(running(lbbcgt_6),861).
orientation(lbbcgt_4,131,862). coords(lbbcgt_4,6,79,862).  visible(lbbcgt_4,862). happensAt(walking(lbbcgt_4),862). orientation(lbbcgt_6,130,862). coords(lbbcgt_6,167,36,862).  visible(lbbcgt_6,862). happensAt(running(lbbcgt_6),862).
orientation(lbbcgt_4,131,863). coords(lbbcgt_4,6,78,863).  visible(lbbcgt_4,863). happensAt(walking(lbbcgt_4),863). orientation(lbbcgt_6,130,863). coords(lbbcgt_6,166,37,863).  visible(lbbcgt_6,863). happensAt(running(lbbcgt_6),863).
orientation(lbbcgt_4,137,864). coords(lbbcgt_4,7,77,864).  visible(lbbcgt_4,864). happensAt(walking(lbbcgt_4),864). orientation(lbbcgt_6,130,864). coords(lbbcgt_6,166,37,864).  visible(lbbcgt_6,864). happensAt(running(lbbcgt_6),864).
orientation(lbbcgt_4,137,865). coords(lbbcgt_4,8,76,865).  visible(lbbcgt_4,865). happensAt(walking(lbbcgt_4),865). orientation(lbbcgt_6,130,865). coords(lbbcgt_6,165,37,865).  visible(lbbcgt_6,865). happensAt(running(lbbcgt_6),865).
orientation(lbbcgt_4,137,866). coords(lbbcgt_4,7,75,866).  visible(lbbcgt_4,866). happensAt(walking(lbbcgt_4),866). orientation(lbbcgt_6,130,866). coords(lbbcgt_6,164,38,866).  visible(lbbcgt_6,866). happensAt(running(lbbcgt_6),866).
orientation(lbbcgt_4,142,867). coords(lbbcgt_4,8,75,867).  visible(lbbcgt_4,867). happensAt(walking(lbbcgt_4),867). orientation(lbbcgt_6,130,867). coords(lbbcgt_6,162,38,867).  visible(lbbcgt_6,867). happensAt(running(lbbcgt_6),867).
orientation(lbbcgt_4,142,868). coords(lbbcgt_4,9,74,868).  visible(lbbcgt_4,868). happensAt(walking(lbbcgt_4),868). orientation(lbbcgt_6,130,868). coords(lbbcgt_6,161,38,868).  visible(lbbcgt_6,868). happensAt(running(lbbcgt_6),868).
orientation(lbbcgt_4,142,869). coords(lbbcgt_4,9,74,869).  visible(lbbcgt_4,869). happensAt(walking(lbbcgt_4),869). orientation(lbbcgt_6,130,869). coords(lbbcgt_6,160,38,869).  visible(lbbcgt_6,869). happensAt(running(lbbcgt_6),869).
orientation(lbbcgt_4,142,870). coords(lbbcgt_4,9,72,870).  visible(lbbcgt_4,870). happensAt(walking(lbbcgt_4),870). orientation(lbbcgt_6,130,870). coords(lbbcgt_6,159,41,870).  visible(lbbcgt_6,870). happensAt(running(lbbcgt_6),870).
orientation(lbbcgt_4,142,871). coords(lbbcgt_4,9,71,871).  visible(lbbcgt_4,871). happensAt(walking(lbbcgt_4),871). orientation(lbbcgt_6,130,871). coords(lbbcgt_6,158,41,871).  visible(lbbcgt_6,871). happensAt(running(lbbcgt_6),871).
orientation(lbbcgt_4,142,872). coords(lbbcgt_4,9,70,872).  visible(lbbcgt_4,872). happensAt(walking(lbbcgt_4),872). orientation(lbbcgt_6,130,872). coords(lbbcgt_6,156,43,872).  visible(lbbcgt_6,872). happensAt(running(lbbcgt_6),872).
orientation(lbbcgt_4,142,873). coords(lbbcgt_4,8,70,873).  visible(lbbcgt_4,873). happensAt(walking(lbbcgt_4),873). orientation(lbbcgt_6,130,873). coords(lbbcgt_6,156,43,873).  visible(lbbcgt_6,873). happensAt(running(lbbcgt_6),873).
orientation(lbbcgt_4,142,874). coords(lbbcgt_4,8,69,874).  visible(lbbcgt_4,874). happensAt(walking(lbbcgt_4),874). orientation(lbbcgt_6,130,874). coords(lbbcgt_6,154,44,874).  visible(lbbcgt_6,874). happensAt(running(lbbcgt_6),874).
orientation(lbbcgt_4,142,875). coords(lbbcgt_4,8,67,875).  visible(lbbcgt_4,875). happensAt(walking(lbbcgt_4),875). orientation(lbbcgt_6,130,875). coords(lbbcgt_6,153,44,875).  visible(lbbcgt_6,875). happensAt(running(lbbcgt_6),875).
orientation(lbbcgt_4,142,876). coords(lbbcgt_4,8,66,876).  visible(lbbcgt_4,876). happensAt(walking(lbbcgt_4),876). orientation(lbbcgt_6,130,876). coords(lbbcgt_6,152,45,876).  visible(lbbcgt_6,876). happensAt(running(lbbcgt_6),876).
orientation(lbbcgt_4,137,877). coords(lbbcgt_4,8,63,877).  visible(lbbcgt_4,877). happensAt(walking(lbbcgt_4),877). orientation(lbbcgt_6,133,877). coords(lbbcgt_6,151,45,877).  visible(lbbcgt_6,877). happensAt(running(lbbcgt_6),877).
orientation(lbbcgt_4,137,878). coords(lbbcgt_4,9,63,878).  visible(lbbcgt_4,878). happensAt(walking(lbbcgt_4),878). orientation(lbbcgt_6,138,878). coords(lbbcgt_6,149,45,878).  visible(lbbcgt_6,878). happensAt(running(lbbcgt_6),878).
orientation(lbbcgt_4,137,879). coords(lbbcgt_4,9,63,879).  visible(lbbcgt_4,879). happensAt(walking(lbbcgt_4),879). orientation(lbbcgt_6,135,879). coords(lbbcgt_6,149,47,879).  visible(lbbcgt_6,879). happensAt(running(lbbcgt_6),879).
orientation(lbbcgt_4,137,880). coords(lbbcgt_4,9,63,880).  visible(lbbcgt_4,880). happensAt(walking(lbbcgt_4),880). orientation(lbbcgt_6,134,880). coords(lbbcgt_6,148,50,880).  visible(lbbcgt_6,880). happensAt(running(lbbcgt_6),880).
orientation(lbbcgt_4,137,881). coords(lbbcgt_4,9,62,881).  visible(lbbcgt_4,881). happensAt(walking(lbbcgt_4),881). orientation(lbbcgt_6,135,881). coords(lbbcgt_6,147,51,881).  visible(lbbcgt_6,881). happensAt(running(lbbcgt_6),881).
#show holdsAt/2.