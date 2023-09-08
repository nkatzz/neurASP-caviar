
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

	orientation(fra1gt_0,145,24). coords(fra1gt_0,195,149,24).  visible(fra1gt_0,24). happensAt(walking(fra1gt_0),24). orientation(fra1gt_1,106,24). coords(fra1gt_1,278,12,24).  visible(fra1gt_1,24). happensAt(walking(fra1gt_1),24).
orientation(fra1gt_0,145,25). coords(fra1gt_0,196,153,25).  visible(fra1gt_0,25). happensAt(walking(fra1gt_0),25). orientation(fra1gt_1,106,25). coords(fra1gt_1,278,12,25).  visible(fra1gt_1,25). happensAt(walking(fra1gt_1),25).
orientation(fra1gt_0,145,26). coords(fra1gt_0,199,154,26).  visible(fra1gt_0,26). happensAt(walking(fra1gt_0),26). orientation(fra1gt_1,106,26). coords(fra1gt_1,278,11,26).  visible(fra1gt_1,26). happensAt(walking(fra1gt_1),26).
orientation(fra1gt_0,145,27). coords(fra1gt_0,201,156,27).  visible(fra1gt_0,27). happensAt(walking(fra1gt_0),27). orientation(fra1gt_1,106,27). coords(fra1gt_1,278,11,27).  visible(fra1gt_1,27). happensAt(walking(fra1gt_1),27).
orientation(fra1gt_0,145,28). coords(fra1gt_0,204,158,28).  visible(fra1gt_0,28). happensAt(walking(fra1gt_0),28). orientation(fra1gt_1,106,28). coords(fra1gt_1,278,11,28).  visible(fra1gt_1,28). happensAt(walking(fra1gt_1),28).
orientation(fra1gt_0,145,29). coords(fra1gt_0,207,159,29).  visible(fra1gt_0,29). happensAt(walking(fra1gt_0),29). orientation(fra1gt_1,106,29). coords(fra1gt_1,278,11,29).  visible(fra1gt_1,29). happensAt(walking(fra1gt_1),29).
orientation(fra1gt_0,145,30). coords(fra1gt_0,208,162,30).  visible(fra1gt_0,30). happensAt(walking(fra1gt_0),30). orientation(fra1gt_1,106,30). coords(fra1gt_1,278,11,30).  visible(fra1gt_1,30). happensAt(walking(fra1gt_1),30).
orientation(fra1gt_0,145,31). coords(fra1gt_0,209,163,31).  visible(fra1gt_0,31). happensAt(walking(fra1gt_0),31). orientation(fra1gt_1,106,31). coords(fra1gt_1,278,11,31).  visible(fra1gt_1,31). happensAt(walking(fra1gt_1),31).
orientation(fra1gt_0,145,32). coords(fra1gt_0,209,165,32).  visible(fra1gt_0,32). happensAt(walking(fra1gt_0),32). orientation(fra1gt_1,103,32). coords(fra1gt_1,278,11,32).  visible(fra1gt_1,32). happensAt(walking(fra1gt_1),32).
orientation(fra1gt_0,156,33). coords(fra1gt_0,209,167,33).  visible(fra1gt_0,33). happensAt(walking(fra1gt_0),33). orientation(fra1gt_1,103,33). coords(fra1gt_1,279,10,33).  visible(fra1gt_1,33). happensAt(walking(fra1gt_1),33).
orientation(fra1gt_0,156,34). coords(fra1gt_0,210,169,34).  visible(fra1gt_0,34). happensAt(walking(fra1gt_0),34). orientation(fra1gt_1,103,34). coords(fra1gt_1,279,8,34).  visible(fra1gt_1,34). happensAt(walking(fra1gt_1),34).
orientation(fra1gt_0,156,35). coords(fra1gt_0,214,171,35).  visible(fra1gt_0,35). happensAt(walking(fra1gt_0),35). orientation(fra1gt_1,103,35). coords(fra1gt_1,279,6,35).  visible(fra1gt_1,35). happensAt(walking(fra1gt_1),35).
orientation(fra1gt_0,156,36). coords(fra1gt_0,217,174,36).  visible(fra1gt_0,36). happensAt(walking(fra1gt_0),36). orientation(fra1gt_1,103,36). coords(fra1gt_1,281,6,36).  visible(fra1gt_1,36). happensAt(walking(fra1gt_1),36).
orientation(fra1gt_0,156,37). coords(fra1gt_0,219,176,37).  visible(fra1gt_0,37). happensAt(walking(fra1gt_0),37). orientation(fra1gt_1,103,37). coords(fra1gt_1,281,6,37).  visible(fra1gt_1,37). happensAt(walking(fra1gt_1),37).
orientation(fra1gt_0,162,38). coords(fra1gt_0,221,178,38).  visible(fra1gt_0,38). happensAt(walking(fra1gt_0),38). orientation(fra1gt_1,103,38). coords(fra1gt_1,281,5,38).  visible(fra1gt_1,38). happensAt(walking(fra1gt_1),38).
orientation(fra1gt_0,162,39). coords(fra1gt_0,222,179,39).  visible(fra1gt_0,39). happensAt(walking(fra1gt_0),39). orientation(fra1gt_1,103,39). coords(fra1gt_1,281,5,39).  visible(fra1gt_1,39). happensAt(walking(fra1gt_1),39).
orientation(fra1gt_0,162,40). coords(fra1gt_0,224,179,40).  visible(fra1gt_0,40). happensAt(walking(fra1gt_0),40). orientation(fra1gt_1,103,40). coords(fra1gt_1,282,5,40).  visible(fra1gt_1,40). happensAt(walking(fra1gt_1),40).
orientation(fra1gt_0,162,41). coords(fra1gt_0,225,182,41).  visible(fra1gt_0,41). happensAt(walking(fra1gt_0),41). orientation(fra1gt_1,103,41). coords(fra1gt_1,282,5,41).  visible(fra1gt_1,41). happensAt(walking(fra1gt_1),41).
orientation(fra1gt_0,162,42). coords(fra1gt_0,227,186,42).  visible(fra1gt_0,42). happensAt(walking(fra1gt_0),42). orientation(fra1gt_1,103,42). coords(fra1gt_1,282,5,42).  visible(fra1gt_1,42). happensAt(walking(fra1gt_1),42).
orientation(fra1gt_0,162,43). coords(fra1gt_0,228,190,43).  visible(fra1gt_0,43). happensAt(walking(fra1gt_0),43). orientation(fra1gt_1,103,43). coords(fra1gt_1,282,5,43).  visible(fra1gt_1,43). happensAt(walking(fra1gt_1),43).
orientation(fra1gt_0,162,44). coords(fra1gt_0,229,193,44).  visible(fra1gt_0,44). happensAt(walking(fra1gt_0),44). orientation(fra1gt_1,103,44). coords(fra1gt_1,282,5,44).  visible(fra1gt_1,44). happensAt(walking(fra1gt_1),44).
orientation(fra1gt_0,167,45). coords(fra1gt_0,230,195,45).  visible(fra1gt_0,45). happensAt(walking(fra1gt_0),45). orientation(fra1gt_1,103,45). coords(fra1gt_1,282,5,45).  visible(fra1gt_1,45). happensAt(walking(fra1gt_1),45).
orientation(fra1gt_0,167,46). coords(fra1gt_0,230,196,46).  visible(fra1gt_0,46). happensAt(walking(fra1gt_0),46). orientation(fra1gt_1,103,46). coords(fra1gt_1,282,4,46).  visible(fra1gt_1,46). happensAt(walking(fra1gt_1),46).
orientation(fra1gt_0,176,47). coords(fra1gt_0,231,197,47).  visible(fra1gt_0,47). happensAt(walking(fra1gt_0),47). orientation(fra1gt_1,103,47). coords(fra1gt_1,282,4,47).  visible(fra1gt_1,47). happensAt(walking(fra1gt_1),47).
orientation(fra1gt_0,6,48). coords(fra1gt_0,232,198,48).  visible(fra1gt_0,48). happensAt(walking(fra1gt_0),48). orientation(fra1gt_1,103,48). coords(fra1gt_1,282,3,48).  visible(fra1gt_1,48). happensAt(walking(fra1gt_1),48).
#show holdsAt/2.