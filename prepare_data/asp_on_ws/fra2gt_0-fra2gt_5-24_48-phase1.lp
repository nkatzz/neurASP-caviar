
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

	orientation(fra2gt_0,159,24). coords(fra2gt_0,175,137,24).  visible(fra2gt_0,24). happensAt(walking(fra2gt_0),24). orientation(fra2gt_5,150,24). coords(fra2gt_5,48,102,24).  visible(fra2gt_5,24). happensAt(walking(fra2gt_5),24).
orientation(fra2gt_0,159,25). coords(fra2gt_0,176,136,25).  visible(fra2gt_0,25). happensAt(walking(fra2gt_0),25). orientation(fra2gt_5,150,25). coords(fra2gt_5,48,102,25).  visible(fra2gt_5,25). happensAt(walking(fra2gt_5),25).
orientation(fra2gt_0,159,26). coords(fra2gt_0,174,136,26).  visible(fra2gt_0,26). happensAt(walking(fra2gt_0),26). orientation(fra2gt_5,150,26). coords(fra2gt_5,47,103,26).  visible(fra2gt_5,26). happensAt(walking(fra2gt_5),26).
orientation(fra2gt_0,159,27). coords(fra2gt_0,173,135,27).  visible(fra2gt_0,27). happensAt(walking(fra2gt_0),27). orientation(fra2gt_5,150,27). coords(fra2gt_5,47,103,27).  visible(fra2gt_5,27). happensAt(walking(fra2gt_5),27).
orientation(fra2gt_0,159,28). coords(fra2gt_0,171,134,28).  visible(fra2gt_0,28). happensAt(walking(fra2gt_0),28). orientation(fra2gt_5,150,28). coords(fra2gt_5,46,104,28).  visible(fra2gt_5,28). happensAt(walking(fra2gt_5),28).
orientation(fra2gt_0,159,29). coords(fra2gt_0,170,134,29).  visible(fra2gt_0,29). happensAt(walking(fra2gt_0),29). orientation(fra2gt_5,150,29). coords(fra2gt_5,46,104,29).  visible(fra2gt_5,29). happensAt(walking(fra2gt_5),29).
orientation(fra2gt_0,159,30). coords(fra2gt_0,170,132,30).  visible(fra2gt_0,30). happensAt(walking(fra2gt_0),30). orientation(fra2gt_5,150,30). coords(fra2gt_5,46,104,30).  visible(fra2gt_5,30). happensAt(walking(fra2gt_5),30).
orientation(fra2gt_0,159,31). coords(fra2gt_0,169,131,31).  visible(fra2gt_0,31). happensAt(walking(fra2gt_0),31). orientation(fra2gt_5,150,31). coords(fra2gt_5,46,104,31).  visible(fra2gt_5,31). happensAt(walking(fra2gt_5),31).
orientation(fra2gt_0,151,32). coords(fra2gt_0,168,129,32).  visible(fra2gt_0,32). happensAt(walking(fra2gt_0),32). orientation(fra2gt_5,150,32). coords(fra2gt_5,45,104,32).  visible(fra2gt_5,32). happensAt(walking(fra2gt_5),32).
orientation(fra2gt_0,149,33). coords(fra2gt_0,166,129,33).  visible(fra2gt_0,33). happensAt(walking(fra2gt_0),33). orientation(fra2gt_5,150,33). coords(fra2gt_5,45,104,33).  visible(fra2gt_5,33). happensAt(walking(fra2gt_5),33).
orientation(fra2gt_0,149,34). coords(fra2gt_0,165,129,34).  visible(fra2gt_0,34). happensAt(walking(fra2gt_0),34). orientation(fra2gt_5,150,34). coords(fra2gt_5,44,104,34).  visible(fra2gt_5,34). happensAt(walking(fra2gt_5),34).
orientation(fra2gt_0,151,35). coords(fra2gt_0,164,129,35).  visible(fra2gt_0,35). happensAt(walking(fra2gt_0),35). orientation(fra2gt_5,150,35). coords(fra2gt_5,44,104,35).  visible(fra2gt_5,35). happensAt(walking(fra2gt_5),35).
orientation(fra2gt_0,154,36). coords(fra2gt_0,163,128,36).  visible(fra2gt_0,36). happensAt(walking(fra2gt_0),36). orientation(fra2gt_5,150,36). coords(fra2gt_5,44,105,36).  visible(fra2gt_5,36). happensAt(walking(fra2gt_5),36).
orientation(fra2gt_0,154,37). coords(fra2gt_0,162,129,37).  visible(fra2gt_0,37). happensAt(active(fra2gt_0),37). orientation(fra2gt_5,150,37). coords(fra2gt_5,44,105,37).  visible(fra2gt_5,37). happensAt(walking(fra2gt_5),37).
orientation(fra2gt_0,154,38). coords(fra2gt_0,162,129,38).  visible(fra2gt_0,38). happensAt(active(fra2gt_0),38). orientation(fra2gt_5,150,38). coords(fra2gt_5,44,105,38).  visible(fra2gt_5,38). happensAt(walking(fra2gt_5),38).
orientation(fra2gt_0,154,39). coords(fra2gt_0,161,129,39).  visible(fra2gt_0,39). happensAt(active(fra2gt_0),39). orientation(fra2gt_5,150,39). coords(fra2gt_5,44,105,39).  visible(fra2gt_5,39). happensAt(walking(fra2gt_5),39).
orientation(fra2gt_0,154,40). coords(fra2gt_0,160,129,40).  visible(fra2gt_0,40). happensAt(active(fra2gt_0),40). orientation(fra2gt_5,150,40). coords(fra2gt_5,43,105,40).  visible(fra2gt_5,40). happensAt(walking(fra2gt_5),40).
orientation(fra2gt_0,154,41). coords(fra2gt_0,159,129,41).  visible(fra2gt_0,41). happensAt(active(fra2gt_0),41). orientation(fra2gt_5,150,41). coords(fra2gt_5,43,105,41).  visible(fra2gt_5,41). happensAt(walking(fra2gt_5),41).
orientation(fra2gt_0,154,42). coords(fra2gt_0,159,129,42).  visible(fra2gt_0,42). happensAt(active(fra2gt_0),42). orientation(fra2gt_5,150,42). coords(fra2gt_5,43,106,42).  visible(fra2gt_5,42). happensAt(walking(fra2gt_5),42).
orientation(fra2gt_0,154,43). coords(fra2gt_0,158,129,43).  visible(fra2gt_0,43). happensAt(active(fra2gt_0),43). orientation(fra2gt_5,150,43). coords(fra2gt_5,43,106,43).  visible(fra2gt_5,43). happensAt(walking(fra2gt_5),43).
orientation(fra2gt_0,154,44). coords(fra2gt_0,158,129,44).  visible(fra2gt_0,44). happensAt(active(fra2gt_0),44). orientation(fra2gt_5,150,44). coords(fra2gt_5,43,106,44).  visible(fra2gt_5,44). happensAt(walking(fra2gt_5),44).
orientation(fra2gt_0,150,45). coords(fra2gt_0,158,129,45).  visible(fra2gt_0,45). happensAt(active(fra2gt_0),45). orientation(fra2gt_5,150,45). coords(fra2gt_5,43,107,45).  visible(fra2gt_5,45). happensAt(walking(fra2gt_5),45).
orientation(fra2gt_0,150,46). coords(fra2gt_0,158,129,46).  visible(fra2gt_0,46). happensAt(active(fra2gt_0),46). orientation(fra2gt_5,150,46). coords(fra2gt_5,41,107,46).  visible(fra2gt_5,46). happensAt(walking(fra2gt_5),46).
orientation(fra2gt_0,150,47). coords(fra2gt_0,158,129,47).  visible(fra2gt_0,47). happensAt(active(fra2gt_0),47). orientation(fra2gt_5,150,47). coords(fra2gt_5,39,106,47).  visible(fra2gt_5,47). happensAt(walking(fra2gt_5),47).
orientation(fra2gt_0,150,48). coords(fra2gt_0,158,129,48).  visible(fra2gt_0,48). happensAt(active(fra2gt_0),48). orientation(fra2gt_5,150,48). coords(fra2gt_5,39,106,48).  visible(fra2gt_5,48). happensAt(walking(fra2gt_5),48).
#show holdsAt/2.