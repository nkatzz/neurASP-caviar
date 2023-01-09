
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
orientation(lbbcgt_4,137,882). coords(lbbcgt_4,9,61,882).  visible(lbbcgt_4,882). happensAt(walking(lbbcgt_4),882). orientation(lbbcgt_6,142,882). coords(lbbcgt_6,148,53,882).  visible(lbbcgt_6,882). happensAt(running(lbbcgt_6),882).
orientation(lbbcgt_4,132,883). coords(lbbcgt_4,9,61,883).  visible(lbbcgt_4,883). happensAt(walking(lbbcgt_4),883). orientation(lbbcgt_6,142,883). coords(lbbcgt_6,147,54,883).  visible(lbbcgt_6,883). happensAt(running(lbbcgt_6),883).
orientation(lbbcgt_4,132,884). coords(lbbcgt_4,10,60,884).  visible(lbbcgt_4,884). happensAt(walking(lbbcgt_4),884). orientation(lbbcgt_6,142,884). coords(lbbcgt_6,146,55,884).  visible(lbbcgt_6,884). happensAt(walking(lbbcgt_6),884).
orientation(lbbcgt_4,132,885). coords(lbbcgt_4,13,59,885).  visible(lbbcgt_4,885). happensAt(walking(lbbcgt_4),885). orientation(lbbcgt_6,142,885). coords(lbbcgt_6,145,56,885).  visible(lbbcgt_6,885). happensAt(walking(lbbcgt_6),885).
orientation(lbbcgt_4,132,886). coords(lbbcgt_4,13,57,886).  visible(lbbcgt_4,886). happensAt(walking(lbbcgt_4),886). orientation(lbbcgt_6,142,886). coords(lbbcgt_6,143,57,886).  visible(lbbcgt_6,886). happensAt(walking(lbbcgt_6),886).
orientation(lbbcgt_4,137,887). coords(lbbcgt_4,13,56,887).  visible(lbbcgt_4,887). happensAt(walking(lbbcgt_4),887). orientation(lbbcgt_6,142,887). coords(lbbcgt_6,142,58,887).  visible(lbbcgt_6,887). happensAt(walking(lbbcgt_6),887).
orientation(lbbcgt_4,137,888). coords(lbbcgt_4,13,55,888).  visible(lbbcgt_4,888). happensAt(walking(lbbcgt_4),888). orientation(lbbcgt_6,142,888). coords(lbbcgt_6,140,58,888).  visible(lbbcgt_6,888). happensAt(walking(lbbcgt_6),888).
orientation(lbbcgt_4,137,889). coords(lbbcgt_4,13,54,889).  visible(lbbcgt_4,889). happensAt(walking(lbbcgt_4),889). orientation(lbbcgt_6,142,889). coords(lbbcgt_6,138,58,889).  visible(lbbcgt_6,889). happensAt(walking(lbbcgt_6),889).
orientation(lbbcgt_4,137,890). coords(lbbcgt_4,13,54,890).  visible(lbbcgt_4,890). happensAt(walking(lbbcgt_4),890). orientation(lbbcgt_6,142,890). coords(lbbcgt_6,136,59,890).  visible(lbbcgt_6,890). happensAt(walking(lbbcgt_6),890).
orientation(lbbcgt_4,137,891). coords(lbbcgt_4,13,53,891).  happensAt(occluded(lbbcgt_4),891). happensAt(walking(lbbcgt_4),891). orientation(lbbcgt_6,142,891). coords(lbbcgt_6,135,60,891).  visible(lbbcgt_6,891). happensAt(walking(lbbcgt_6),891).
orientation(lbbcgt_4,137,892). coords(lbbcgt_4,13,53,892).  happensAt(occluded(lbbcgt_4),892). happensAt(walking(lbbcgt_4),892). orientation(lbbcgt_6,142,892). coords(lbbcgt_6,134,63,892).  visible(lbbcgt_6,892). happensAt(walking(lbbcgt_6),892).
orientation(lbbcgt_4,137,893). coords(lbbcgt_4,13,53,893).  happensAt(occluded(lbbcgt_4),893). happensAt(walking(lbbcgt_4),893). orientation(lbbcgt_6,142,893). coords(lbbcgt_6,134,64,893).  visible(lbbcgt_6,893). happensAt(walking(lbbcgt_6),893).
#show holdsAt/2.