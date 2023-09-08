
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

	orientation(ms3ggt_3,162,858). coords(ms3ggt_3,33,172,858).  visible(ms3ggt_3,858). happensAt(walking(ms3ggt_3),858). orientation(ms3ggt_4,140,858). coords(ms3ggt_4,133,40,858).  visible(ms3ggt_4,858). happensAt(walking(ms3ggt_4),858).
orientation(ms3ggt_3,162,859). coords(ms3ggt_3,33,173,859).  visible(ms3ggt_3,859). happensAt(walking(ms3ggt_3),859). orientation(ms3ggt_4,140,859). coords(ms3ggt_4,133,40,859).  visible(ms3ggt_4,859). happensAt(walking(ms3ggt_4),859).
orientation(ms3ggt_3,162,860). coords(ms3ggt_3,33,173,860).  visible(ms3ggt_3,860). happensAt(walking(ms3ggt_3),860). orientation(ms3ggt_4,140,860). coords(ms3ggt_4,133,40,860).  visible(ms3ggt_4,860). happensAt(walking(ms3ggt_4),860).
orientation(ms3ggt_3,162,861). coords(ms3ggt_3,33,175,861).  visible(ms3ggt_3,861). happensAt(walking(ms3ggt_3),861). orientation(ms3ggt_4,140,861). coords(ms3ggt_4,133,40,861).  visible(ms3ggt_4,861). happensAt(walking(ms3ggt_4),861).
orientation(ms3ggt_3,162,862). coords(ms3ggt_3,33,175,862).  visible(ms3ggt_3,862). happensAt(walking(ms3ggt_3),862). orientation(ms3ggt_4,140,862). coords(ms3ggt_4,132,41,862).  visible(ms3ggt_4,862). happensAt(walking(ms3ggt_4),862).
orientation(ms3ggt_3,162,863). coords(ms3ggt_3,33,177,863).  visible(ms3ggt_3,863). happensAt(walking(ms3ggt_3),863). orientation(ms3ggt_4,140,863). coords(ms3ggt_4,132,41,863).  visible(ms3ggt_4,863). happensAt(walking(ms3ggt_4),863).
orientation(ms3ggt_3,162,864). coords(ms3ggt_3,33,179,864).  visible(ms3ggt_3,864). happensAt(walking(ms3ggt_3),864). orientation(ms3ggt_4,140,864). coords(ms3ggt_4,132,41,864).  visible(ms3ggt_4,864). happensAt(walking(ms3ggt_4),864).
orientation(ms3ggt_3,162,865). coords(ms3ggt_3,33,179,865).  visible(ms3ggt_3,865). happensAt(walking(ms3ggt_3),865). orientation(ms3ggt_4,140,865). coords(ms3ggt_4,132,41,865).  visible(ms3ggt_4,865). happensAt(walking(ms3ggt_4),865).
orientation(ms3ggt_3,162,866). coords(ms3ggt_3,33,180,866).  visible(ms3ggt_3,866). happensAt(walking(ms3ggt_3),866). orientation(ms3ggt_4,140,866). coords(ms3ggt_4,132,41,866).  visible(ms3ggt_4,866). happensAt(walking(ms3ggt_4),866).
orientation(ms3ggt_3,162,867). coords(ms3ggt_3,33,181,867).  visible(ms3ggt_3,867). happensAt(walking(ms3ggt_3),867). orientation(ms3ggt_4,140,867). coords(ms3ggt_4,132,41,867).  visible(ms3ggt_4,867). happensAt(walking(ms3ggt_4),867).
orientation(ms3ggt_3,162,868). coords(ms3ggt_3,33,182,868).  visible(ms3ggt_3,868). happensAt(walking(ms3ggt_3),868). orientation(ms3ggt_4,140,868). coords(ms3ggt_4,132,41,868).  visible(ms3ggt_4,868). happensAt(walking(ms3ggt_4),868).
orientation(ms3ggt_3,162,869). coords(ms3ggt_3,33,183,869).  visible(ms3ggt_3,869). happensAt(walking(ms3ggt_3),869). orientation(ms3ggt_4,140,869). coords(ms3ggt_4,131,42,869).  visible(ms3ggt_4,869). happensAt(walking(ms3ggt_4),869).
orientation(ms3ggt_3,162,870). coords(ms3ggt_3,33,184,870).  visible(ms3ggt_3,870). happensAt(walking(ms3ggt_3),870). orientation(ms3ggt_4,140,870). coords(ms3ggt_4,131,42,870).  visible(ms3ggt_4,870). happensAt(walking(ms3ggt_4),870).
orientation(ms3ggt_3,170,871). coords(ms3ggt_3,33,184,871).  visible(ms3ggt_3,871). happensAt(walking(ms3ggt_3),871). orientation(ms3ggt_4,140,871). coords(ms3ggt_4,130,42,871).  visible(ms3ggt_4,871). happensAt(walking(ms3ggt_4),871).
orientation(ms3ggt_3,170,872). coords(ms3ggt_3,33,185,872).  visible(ms3ggt_3,872). happensAt(walking(ms3ggt_3),872). orientation(ms3ggt_4,140,872). coords(ms3ggt_4,130,42,872).  visible(ms3ggt_4,872). happensAt(walking(ms3ggt_4),872).
orientation(ms3ggt_3,170,873). coords(ms3ggt_3,32,187,873).  visible(ms3ggt_3,873). happensAt(walking(ms3ggt_3),873). orientation(ms3ggt_4,140,873). coords(ms3ggt_4,130,42,873).  visible(ms3ggt_4,873). happensAt(walking(ms3ggt_4),873).
orientation(ms3ggt_3,170,874). coords(ms3ggt_3,32,189,874).  visible(ms3ggt_3,874). happensAt(walking(ms3ggt_3),874). orientation(ms3ggt_4,140,874). coords(ms3ggt_4,129,42,874).  visible(ms3ggt_4,874). happensAt(walking(ms3ggt_4),874).
orientation(ms3ggt_3,170,875). coords(ms3ggt_3,32,190,875).  visible(ms3ggt_3,875). happensAt(walking(ms3ggt_3),875). orientation(ms3ggt_4,140,875). coords(ms3ggt_4,129,42,875).  visible(ms3ggt_4,875). happensAt(walking(ms3ggt_4),875).
orientation(ms3ggt_3,170,876). coords(ms3ggt_3,32,193,876).  visible(ms3ggt_3,876). happensAt(walking(ms3ggt_3),876). orientation(ms3ggt_4,140,876). coords(ms3ggt_4,129,43,876).  visible(ms3ggt_4,876). happensAt(walking(ms3ggt_4),876).
orientation(ms3ggt_3,170,877). coords(ms3ggt_3,32,193,877).  visible(ms3ggt_3,877). happensAt(walking(ms3ggt_3),877). orientation(ms3ggt_4,140,877). coords(ms3ggt_4,129,43,877).  visible(ms3ggt_4,877). happensAt(walking(ms3ggt_4),877).
orientation(ms3ggt_3,170,878). coords(ms3ggt_3,32,193,878).  visible(ms3ggt_3,878). happensAt(walking(ms3ggt_3),878). orientation(ms3ggt_4,140,878). coords(ms3ggt_4,128,43,878).  visible(ms3ggt_4,878). happensAt(walking(ms3ggt_4),878).
orientation(ms3ggt_3,162,879). coords(ms3ggt_3,32,196,879).  visible(ms3ggt_3,879). happensAt(walking(ms3ggt_3),879). orientation(ms3ggt_4,140,879). coords(ms3ggt_4,128,43,879).  visible(ms3ggt_4,879). happensAt(walking(ms3ggt_4),879).
orientation(ms3ggt_3,162,880). coords(ms3ggt_3,31,197,880).  visible(ms3ggt_3,880). happensAt(walking(ms3ggt_3),880). orientation(ms3ggt_4,140,880). coords(ms3ggt_4,128,44,880).  visible(ms3ggt_4,880). happensAt(walking(ms3ggt_4),880).
orientation(ms3ggt_3,162,881). coords(ms3ggt_3,31,197,881).  visible(ms3ggt_3,881). happensAt(walking(ms3ggt_3),881). orientation(ms3ggt_4,140,881). coords(ms3ggt_4,128,44,881).  visible(ms3ggt_4,881). happensAt(walking(ms3ggt_4),881).
orientation(ms3ggt_3,162,882). coords(ms3ggt_3,32,197,882).  visible(ms3ggt_3,882). happensAt(walking(ms3ggt_3),882). orientation(ms3ggt_4,140,882). coords(ms3ggt_4,128,44,882).  visible(ms3ggt_4,882). happensAt(walking(ms3ggt_4),882).
#show holdsAt/2.