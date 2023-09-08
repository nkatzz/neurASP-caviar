
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
orientation(ms3ggt_3,162,883). coords(ms3ggt_3,32,198,883).  visible(ms3ggt_3,883). happensAt(walking(ms3ggt_3),883). orientation(ms3ggt_4,140,883). coords(ms3ggt_4,128,44,883).  visible(ms3ggt_4,883). happensAt(walking(ms3ggt_4),883).
orientation(ms3ggt_3,170,884). coords(ms3ggt_3,32,200,884).  visible(ms3ggt_3,884). happensAt(walking(ms3ggt_3),884). orientation(ms3ggt_4,140,884). coords(ms3ggt_4,128,44,884).  visible(ms3ggt_4,884). happensAt(walking(ms3ggt_4),884).
orientation(ms3ggt_3,170,885). coords(ms3ggt_3,32,202,885).  visible(ms3ggt_3,885). happensAt(walking(ms3ggt_3),885). orientation(ms3ggt_4,140,885). coords(ms3ggt_4,128,44,885).  visible(ms3ggt_4,885). happensAt(walking(ms3ggt_4),885).
orientation(ms3ggt_3,170,886). coords(ms3ggt_3,32,203,886).  visible(ms3ggt_3,886). happensAt(walking(ms3ggt_3),886). orientation(ms3ggt_4,140,886). coords(ms3ggt_4,128,44,886).  visible(ms3ggt_4,886). happensAt(walking(ms3ggt_4),886).
orientation(ms3ggt_3,170,887). coords(ms3ggt_3,32,204,887).  visible(ms3ggt_3,887). happensAt(walking(ms3ggt_3),887). orientation(ms3ggt_4,140,887). coords(ms3ggt_4,127,44,887).  visible(ms3ggt_4,887). happensAt(walking(ms3ggt_4),887).
orientation(ms3ggt_3,170,888). coords(ms3ggt_3,32,205,888).  visible(ms3ggt_3,888). happensAt(walking(ms3ggt_3),888). orientation(ms3ggt_4,140,888). coords(ms3ggt_4,127,44,888).  visible(ms3ggt_4,888). happensAt(walking(ms3ggt_4),888).
orientation(ms3ggt_3,170,889). coords(ms3ggt_3,32,205,889).  visible(ms3ggt_3,889). happensAt(walking(ms3ggt_3),889). orientation(ms3ggt_4,140,889). coords(ms3ggt_4,127,45,889).  visible(ms3ggt_4,889). happensAt(walking(ms3ggt_4),889).
orientation(ms3ggt_3,170,890). coords(ms3ggt_3,32,206,890).  visible(ms3ggt_3,890). happensAt(walking(ms3ggt_3),890). orientation(ms3ggt_4,140,890). coords(ms3ggt_4,127,45,890).  visible(ms3ggt_4,890). happensAt(walking(ms3ggt_4),890).
orientation(ms3ggt_3,170,891). coords(ms3ggt_3,31,207,891).  visible(ms3ggt_3,891). happensAt(walking(ms3ggt_3),891). orientation(ms3ggt_4,140,891). coords(ms3ggt_4,127,45,891).  visible(ms3ggt_4,891). happensAt(walking(ms3ggt_4),891).
orientation(ms3ggt_3,170,892). coords(ms3ggt_3,31,207,892).  visible(ms3ggt_3,892). happensAt(walking(ms3ggt_3),892). orientation(ms3ggt_4,140,892). coords(ms3ggt_4,126,45,892).  visible(ms3ggt_4,892). happensAt(walking(ms3ggt_4),892).
orientation(ms3ggt_3,170,893). coords(ms3ggt_3,32,207,893).  visible(ms3ggt_3,893). happensAt(walking(ms3ggt_3),893). orientation(ms3ggt_4,140,893). coords(ms3ggt_4,126,45,893).  visible(ms3ggt_4,893). happensAt(walking(ms3ggt_4),893).
orientation(ms3ggt_3,170,894). coords(ms3ggt_3,29,208,894).  visible(ms3ggt_3,894). happensAt(walking(ms3ggt_3),894). orientation(ms3ggt_4,140,894). coords(ms3ggt_4,126,45,894).  visible(ms3ggt_4,894). happensAt(walking(ms3ggt_4),894).
#show holdsAt/2.